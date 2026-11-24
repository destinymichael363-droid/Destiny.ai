from __future__ import annotations

import json
import os
import signal
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from broker_and_account import VirtualAccount
from ob_fvg_engine import run_ob_fvg_engine
from sr_engine import run_sr_zones_engine

try:
    from strategy_engine import evaluate_setups as _evaluate_setups  # type: ignore
except Exception:
    _evaluate_setups = None

try:
    import mtf_runner as _mtf_runner  # type: ignore
except Exception:
    _mtf_runner = None

STATE_PATH = Path("data/worker_state.json")
VA_DIR = Path("data/virtual_accounts")
POLL_INTERVAL_SEC = int(os.getenv("WORKER_POLL_SEC", "30"))
MAX_SIGNALS = int(os.getenv("WORKER_MAX_SIGNALS", "25"))
MAX_AUTO_OPEN_POSITIONS = int(os.getenv("MAX_AUTO_OPEN_POSITIONS", "3"))

WORKER_WATCHLIST: list[tuple[str, str]] = [
    ("Gold (XAU)", "GC=F"),
    ("Crude Oil (WTI)", "CL=F"),
    ("EUR/USD", "EURUSD=X"),
    ("GBP/USD", "GBPUSD=X"),
    ("Bitcoin (BTC)", "BTC-USD"),
]

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
_RUNNING = True


def _handle_stop(signum: int, frame: object) -> None:
    _ = (signum, frame)
    global _RUNNING
    _RUNNING = False


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _get_val(obj: object, names: tuple[str, ...], default: object = None) -> object:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _supabase_config() -> tuple[str, str]:
    url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    key = (
        os.getenv("SUPABASE_KEY", "").strip()
        or os.getenv("SUPABASE_ANON_KEY", "").strip()
        or os.getenv("SUPABASE_SERVICE_KEY", "").strip()
    )
    return url, key


def _supabase_load_worker_state() -> dict | None:
    url, key = _supabase_config()
    if not url or not key:
        return None
    endpoint = f"{url}/rest/v1/worker_state?id=eq.default&select=payload&limit=1"
    req = urllib.request.Request(
        endpoint,
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            rows = json.loads(resp.read().decode("utf-8"))
            if isinstance(rows, list) and rows:
                raw_payload = rows[0].get("payload")
                if isinstance(raw_payload, str):
                    return json.loads(raw_payload)
                if isinstance(raw_payload, dict):
                    return raw_payload
    except Exception:
        return None
    return None


def _supabase_save_worker_state(state: dict) -> bool:
    url, key = _supabase_config()
    if not url or not key:
        return False
    endpoint = f"{url}/rest/v1/worker_state?on_conflict=id"
    body = json.dumps([{"id": "default", "payload": state}]).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, Exception):
        return False


def _supabase_load_account(username: str) -> dict | None:
    url, key = _supabase_config()
    if not url or not key:
        return None
    endpoint = f"{url}/rest/v1/virtual_accounts?username=eq.{urllib.parse.quote(username)}&select=payload&limit=1"
    req = urllib.request.Request(
        endpoint,
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            rows = json.loads(resp.read().decode("utf-8"))
            if isinstance(rows, list) and rows:
                raw_payload = rows[0].get("payload")
                if isinstance(raw_payload, str):
                    return json.loads(raw_payload)
                if isinstance(raw_payload, dict):
                    return raw_payload
    except Exception:
        return None
    return None


def _supabase_save_account(username: str, payload: dict) -> bool:
    url, key = _supabase_config()
    if not url or not key:
        return False
    endpoint = f"{url}/rest/v1/virtual_accounts?on_conflict=username"
    body = json.dumps([{"username": username, "payload": payload}]).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, Exception):
        return False


def _va_file_path(username: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in username)[:40] or "default"
    VA_DIR.mkdir(parents=True, exist_ok=True)
    return VA_DIR / f"{safe_name}.json"


def _load_raw_account_payload(username: str) -> dict:
    acct_path = _va_file_path(username)
    remote = _supabase_load_account(username)
    if isinstance(remote, dict):
        return remote
    if acct_path.exists():
        try:
            local = json.loads(acct_path.read_text(encoding="utf-8"))
            if isinstance(local, dict):
                return local
        except Exception:
            pass
    return {
        "balance": 10000.0,
        "positions": [],
        "closed_positions": [],
        "auto_exec_virtual": True,
        "risk_pct": 2.0,
    }


def _account_to_payload(acct: VirtualAccount, closed_log: list[dict], prefs: dict | None = None) -> dict:
    open_list = []
    for pos in list(getattr(acct, "positions", []) or []):
        sz_val = float(_get_val(pos, ("size", "qty", "quantity", "units"), 1.0))
        units_val = float(_get_val(pos, ("units", "size", "qty", "quantity"), sz_val))
        entry_v = float(_get_val(pos, ("entry_price", "entry"), 0.0))
        margin_v = float(_get_val(pos, ("margin_used", "margin", "collateral"), max(0.0, abs(units_val * entry_v) * 0.05)))
        open_list.append({
            "symbol": str(_get_val(pos, ("symbol",), "")),
            "side": str(_get_val(pos, ("side",), "LONG")),
            "entry_price": entry_v,
            "stop_loss": float(_get_val(pos, ("stop_loss", "sl"), 0.0)),
            "take_profit": float(_get_val(pos, ("take_profit", "tp"), 0.0)),
            "size": sz_val,
            "units": units_val,
            "margin_used": margin_v,
        })
    p = prefs or {}
    return {
        "balance": round(float(getattr(acct, "balance", 10000.0)), 2),
        "positions": open_list,
        "closed_positions": list(closed_log or []),
        "auto_exec_virtual": bool(p.get("auto_exec_virtual", True)),
        "risk_pct": float(p.get("risk_pct", 2.0)),
    }


def _payload_to_account(data: dict) -> tuple[VirtualAccount, list[dict], dict]:
    acct = VirtualAccount()
    closed_log: list[dict] = []
    prefs = {"auto_exec_virtual": True, "risk_pct": 2.0}
    if not isinstance(data, dict):
        return acct, closed_log, prefs

    prefs["auto_exec_virtual"] = bool(data.get("auto_exec_virtual", True))
    prefs["risk_pct"] = float(data.get("risk_pct", 2.0))

    if "balance" in data and hasattr(acct, "balance"):
        acct.balance = round(float(data["balance"]), 2)
    closed_log = list(data.get("closed_positions") or [])
    saved_positions = list(data.get("positions") or [])
    if saved_positions and hasattr(acct, "positions") and isinstance(acct.positions, list):
        for item in saved_positions:
            sym = str(item.get("symbol", ""))
            side = str(item.get("side", "LONG"))
            ent = float(item.get("entry_price", 0.0))
            sl = float(item.get("stop_loss", 0.0))
            tp = float(item.get("take_profit", 0.0))
            sz = float(item.get("size", item.get("units", 1.0)))
            raw_units = float(item.get("units", sz))
            margin_used = float(item.get("margin_used", max(0.0, abs(raw_units * ent) * 0.05)))
            try:
                acct.positions.append(
                    type("SavedPos", (), {
                        "symbol": sym,
                        "side": side,
                        "entry_price": ent,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "size": sz,
                        "units": raw_units,
                        "margin_used": margin_used,
                    })()
                )
            except Exception:
                pass
    return acct, closed_log, prefs


def _load_virtual_account_obj(username: str) -> tuple[VirtualAccount, list[dict], dict]:
    raw = _load_raw_account_payload(username)
    return _payload_to_account(raw)


def _save_virtual_account_obj(username: str, acct: VirtualAccount, closed_log: list[dict], prefs: dict | None = None) -> None:
    payload = _account_to_payload(acct, closed_log, prefs=prefs)
    try:
        _va_file_path(username).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass
    _supabase_save_account(username, payload)


def _discover_trader_usernames() -> list[str]:
    users: set[str] = set()
    env_users = os.getenv("AUTO_EXEC_USERS", "").strip()
    if env_users:
        for u in env_users.split(","):
            clean = u.strip()
            if clean:
                users.add(clean)
    single_user = os.getenv("AUTO_EXEC_USER", "").strip()
    if single_user:
        users.add(single_user)
    if VA_DIR.exists():
        for p in VA_DIR.glob("*.json"):
            if p.stem:
                users.add(p.stem)
    return sorted(users)


def _open_via_virtual_account(
    acct: VirtualAccount,
    symbol: str,
    side: str,
    entry_px: float,
    sl_px: float,
    tp_px: float,
    risk_pct: float,
) -> tuple[bool, str]:
    pre_balance = round(float(getattr(acct, "balance", 10000.0)), 2)
    intended_risk_dollars = round(pre_balance * (float(risk_pct) / 100.0), 2)

    open_fn = getattr(acct, "open_position_with_risk", None)
    if callable(open_fn):
        for call in (
            lambda: open_fn(symbol, side, entry_px, sl_px, tp_px, risk_pct=risk_pct),
            lambda: open_fn(symbol, side, entry_px, sl_px, tp_px, risk_pct),
            lambda: open_fn(symbol, side, entry_px, sl_px, tp_px),
        ):
            try:
                res = call()
                if isinstance(res, tuple) and len(res) >= 2:
                    ok, msg = bool(res[0]), str(res[1])
                else:
                    ok, msg = bool(res), "Opened via VirtualAccount"
                if ok and hasattr(acct, "balance"):
                    deducted = pre_balance - float(acct.balance)
                    if intended_risk_dollars > 0 and abs(deducted - intended_risk_dollars) > 0.05:
                        acct.balance = round(pre_balance - intended_risk_dollars, 2)
                return ok, msg
            except TypeError:
                continue

    sl_dist = abs(entry_px - sl_px)
    if sl_dist <= 0 or intended_risk_dollars <= 0 or pre_balance < intended_risk_dollars:
        return False, "signal_only_risk_guard"
    units = round(intended_risk_dollars / sl_dist, 6)
    positions = list(getattr(acct, "positions", []) or [])
    positions.append(
        type("WorkerPos", (), {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_px,
            "stop_loss": sl_px,
            "take_profit": tp_px,
            "size": units,
            "units": units,
            "margin_used": round(abs(units * entry_px) * 0.05, 2),
        })()
    )
    acct.positions = positions
    acct.balance = round(pre_balance - intended_risk_dollars, 2)
    return True, "opened_fallback"


def _evaluate_tp_sl_hits_for_users(
    latest_prices: dict[str, tuple[float, float, float]],
    usernames: list[str],
) -> int:
    if not latest_prices or not usernames:
        return 0

    total_closed = 0
    for username in usernames:
        acct, closed_log, prefs = _load_virtual_account_obj(username)
        positions = list(getattr(acct, "positions", []) or [])
        if not positions:
            continue

        remaining = []
        closed_count = 0
        for pos in positions:
            sym = str(_get_val(pos, ("symbol",), ""))
            if sym not in latest_prices:
                remaining.append(pos)
                continue

            bar_hi, bar_lo, bar_close = latest_prices[sym]
            side = str(_get_val(pos, ("side",), "LONG")).upper()
            entry_p = float(_get_val(pos, ("entry_price", "entry"), bar_close))
            sl_p = float(_get_val(pos, ("stop_loss", "sl"), 0.0))
            tp_p = float(_get_val(pos, ("take_profit", "tp"), 0.0))
            units_val = float(_get_val(pos, ("units", "size", "qty", "quantity"), 0.0))

            exit_px: float | None = None
            if side == "LONG":
                if sl_p > 0 and bar_lo <= sl_p:
                    exit_px = sl_p
                elif tp_p > 0 and bar_hi >= tp_p:
                    exit_px = tp_p
            elif side == "SHORT":
                if sl_p > 0 and bar_hi >= sl_p:
                    exit_px = sl_p
                elif tp_p > 0 and bar_lo <= tp_p:
                    exit_px = tp_p

            if exit_px is None:
                remaining.append(pos)
                continue

            signed_units = -abs(units_val) if side == "SHORT" else abs(units_val)
            pnl_val = round(float((exit_px - entry_p) * signed_units), 2)
            if hasattr(acct, "balance"):
                acct.balance = round(float(acct.balance) + pnl_val, 2)

            closed_log.append({
                "symbol": sym,
                "side": side,
                "entry": entry_p,
                "exit": float(exit_px),
                "sl": sl_p,
                "tp": tp_p,
                "pnl": pnl_val,
                "closed_by": "worker_tp_sl",
            })
            closed_count += 1

        if closed_count > 0:
            acct.positions = remaining
            _save_virtual_account_obj(username, acct, closed_log, prefs=prefs)
            total_closed += closed_count

    return total_closed


def _try_auto_execute_for_users(signal_item: dict, usernames: list[str], default_risk_pct: float = 2.0) -> str:
    if not usernames:
        return "auto_ready_no_trader"

    entry_px = float(signal_item.get("entry", 0) or 0)
    sl_px = float(signal_item.get("sl", 0) or 0)
    tp_px = float(signal_item.get("tp", 0) or 0)
    side = str(signal_item.get("direction", "")).upper()
    symbol = str(signal_item.get("symbol", "")).strip()
    if entry_px <= 0 or sl_px <= 0 or tp_px <= 0 or side not in ("LONG", "SHORT") or not symbol:
        return "signal_only_invalid_levels"

    executed_users: list[str] = []
    for username in usernames:
        acct, closed_log, prefs = _load_virtual_account_obj(username)
        if not bool(prefs.get("auto_exec_virtual", True)):
            continue
        positions = list(getattr(acct, "positions", []) or [])
        if len(positions) >= MAX_AUTO_OPEN_POSITIONS:
            continue
        user_risk = float(prefs.get("risk_pct", default_risk_pct) or default_risk_pct)
        ok, _msg = _open_via_virtual_account(acct, symbol, side, entry_px, sl_px, tp_px, risk_pct=user_risk)
        if ok:
            _save_virtual_account_obj(username, acct, closed_log, prefs=prefs)
            executed_users.append(username)

    if executed_users:
        return f"virtual_executed ({', '.join(executed_users)})"
    return "auto_armed_no_new_fill"


def _clean_ohlcv(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV)
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    if not all(c in out.columns for c in ["Open", "High", "Low", "Close"]):
        return pd.DataFrame(columns=OHLCV)
    if "Volume" not in out.columns:
        out["Volume"] = 0.0
    out = out[OHLCV].copy()
    for col in OHLCV:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["Open", "High", "Low", "Close"]).sort_index().tail(200)


def _resample_4h(hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly is None or hourly.empty:
        return pd.DataFrame(columns=OHLCV)
    try:
        res = hourly.resample("4h", label="right", closed="right").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })
        return _clean_ohlcv(res)
    except Exception:
        return pd.DataFrame(columns=OHLCV)


def _download_tf(symbol: str, interval: str, period: str, resample_4h: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(columns=OHLCV)
    try:
        dl = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )
        df = _clean_ohlcv(dl)
    except Exception:
        df = pd.DataFrame(columns=OHLCV)

    if df.empty:
        try:
            raw = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False, actions=False)
            df = _clean_ohlcv(raw)
        except Exception:
            df = pd.DataFrame(columns=OHLCV)

    if resample_4h and not df.empty:
        df = _resample_4h(df)
    return df


def _load_existing_state() -> dict:
    if STATE_PATH.exists():
        try:
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("signals", [])
                data.setdefault("seen_keys", [])
                return data
        except Exception:
            pass
    remote = _supabase_load_worker_state()
    if isinstance(remote, dict):
        remote.setdefault("signals", [])
        remote.setdefault("seen_keys", [])
        return remote
    return {
        "status": "starting",
        "updated_utc": _utc_now_str(),
        "cycle_count": 0,
        "scanned_assets": [],
        "signals": [],
        "seen_keys": [],
    }


def _write_state_atomic(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = STATE_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp_path, STATE_PATH)
    _supabase_save_worker_state(state)


def _call_evaluate_setups_mtf(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    symbol: str,
    sr_out: dict,
    ob_out: dict,
    mtf_context: dict,
    rr_target: float = 5.0,
) -> list:
    if _evaluate_setups is None or df_15m.empty:
        return []

    attempts = [
        lambda: _evaluate_setups(
            df=df_15m,
            symbol=symbol,
            tf_label="15M",
            sr_out=sr_out,
            ob_out=ob_out,
            rr_target=rr_target,
            df_1h=df_1h,
            df_4h=df_4h,
            mtf_context=mtf_context,
        ),
        lambda: _evaluate_setups(
            df=df_15m,
            symbol=symbol,
            tf_label="15M",
            sr_out=sr_out,
            ob_out=ob_out,
            rr_target=rr_target,
            mtf_data={"15M": df_15m, "1H": df_1h, "4H": df_4h, **mtf_context},
        ),
        lambda: _evaluate_setups(
            df=df_15m,
            symbol=symbol,
            tf_label="15M",
            sr_out=sr_out,
            ob_out=ob_out,
            rr_target=rr_target,
        ),
    ]
    for fn in attempts:
        try:
            res = fn()
            if res is not None:
                return list(res)
        except TypeError:
            continue
        except Exception:
            return []
    return []


def _scan_once(state: dict, risk_pct: float = 2.0, target_users: list[str] | None = None) -> dict:
    signals: list[dict] = list(state.get("signals") or [])
    seen_keys: list[str] = list(state.get("seen_keys") or [])
    seen_set = set(seen_keys)
    scanned: list[str] = []
    latest_prices: dict[str, tuple[float, float, float]] = {}

    users_to_process = target_users if target_users is not None else _discover_trader_usernames()

    for asset_name, symbol in WORKER_WATCHLIST:
        df_15m = _download_tf(symbol, interval="15m", period="14d", resample_4h=False)
        if df_15m.empty:
            continue

        latest_prices[symbol] = (
            float(df_15m["High"].iloc[-1]),
            float(df_15m["Low"].iloc[-1]),
            float(df_15m["Close"].iloc[-1]),
        )
        scanned.append(f"{symbol}@{str(df_15m.index[-1])}")

        if _evaluate_setups is None:
            continue

        df_1h = _download_tf(symbol, interval="1h", period="60d", resample_4h=False)
        df_4h = _download_tf(symbol, interval="1h", period="120d", resample_4h=True)

        try:
            sr_out = run_sr_zones_engine(df_15m, tf_label="15M", invalidation_mode="Wick")
        except Exception:
            sr_out = {"active_zones": [], "minor_zones": []}

        try:
            ob_out = run_ob_fvg_engine(df_15m, timeframe_minutes=15)
        except Exception:
            ob_out = {}

        mtf_context: dict = {"15M": df_15m, "1H": df_1h, "4H": df_4h}
        if _mtf_runner is not None:
            for hook_name in ("build_mtf_context", "get_mtf_context", "enrich_chart_context"):
                fn = getattr(_mtf_runner, hook_name, None)
                if callable(fn):
                    try:
                        out = fn(
                            symbol=symbol,
                            tf_label="15M",
                            df=df_15m,
                            df_1h=df_1h,
                            df_4h=df_4h,
                            sr_out=sr_out,
                            ob_out=ob_out,
                        )
                        if isinstance(out, dict):
                            sr_out = out.get("sr_out", sr_out)
                            ob_out = out.get("ob_out", ob_out)
                            mtf_context.update(out)
                        break
                    except TypeError:
                        continue
                    except Exception:
                        break

        setups = _call_evaluate_setups_mtf(
            df_15m=df_15m,
            df_1h=df_1h,
            df_4h=df_4h,
            symbol=symbol,
            sr_out=sr_out,
            ob_out=ob_out,
            mtf_context=mtf_context,
            rr_target=5.0,
        )

        bar_ts = str(df_15m.index[-1])
        for s in setups:
            direction = str(_get_val(s, ("direction",), "SIGNAL"))
            conf = str(_get_val(s, ("confidence",), "15M/1H/4H"))
            entry = _get_val(s, ("entry",), "n/a")
            sl = _get_val(s, ("sl",), "n/a")
            tp = _get_val(s, ("tp",), "n/a")
            rr = _get_val(s, ("rr",), 5.0)

            dedupe_key = f"{symbol}|15M|{direction}|{bar_ts}|{entry}"
            if dedupe_key in seen_set:
                continue
            seen_set.add(dedupe_key)
            seen_keys.append(dedupe_key)

            sig_item = {
                "asset": asset_name,
                "symbol": symbol,
                "direction": direction,
                "confidence": conf,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "rr": rr,
                "bar_time": bar_ts,
                "logged_utc": _utc_now_str(),
                "execution_status": "auto_armed",
            }
            sig_item["execution_status"] = _try_auto_execute_for_users(
                sig_item, usernames=users_to_process, default_risk_pct=risk_pct
            )
            signals.append(sig_item)

    closed_hits = _evaluate_tp_sl_hits_for_users(latest_prices, usernames=users_to_process)
    state["last_tp_sl_closed"] = closed_hits
    state["scanned_assets"] = scanned
    state["signals"] = signals[-MAX_SIGNALS:]
    state["seen_keys"] = seen_keys[-200:]
    return state


def run_worker_scan_once(
    status_label: str = "auto_scan_ok",
    risk_pct: float = 2.0,
    target_users: list[str] | None = None,
) -> dict:
    state = _load_existing_state()
    try:
        state["cycle_count"] = int(state.get("cycle_count", 0)) + 1
        state = _scan_once(state, risk_pct=risk_pct, target_users=target_users)
        state["status"] = status_label
        state["updated_utc"] = _utc_now_str()
        _write_state_atomic(state)
    except Exception as exc:
        state["status"] = f"error: {exc}"
        state["updated_utc"] = _utc_now_str()
        try:
            _write_state_atomic(state)
        except Exception:
            pass
    return state


def run_worker_loop() -> None:
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    state = _load_existing_state()
    state["status"] = "running"
    state["updated_utc"] = _utc_now_str()
    _write_state_atomic(state)

    while _RUNNING:
        try:
            state["cycle_count"] = int(state.get("cycle_count", 0)) + 1
            state = _scan_once(state)
            state["status"] = "running"
            state["updated_utc"] = _utc_now_str()
            _write_state_atomic(state)
        except Exception as exc:
            state["status"] = f"error: {exc}"
            state["updated_utc"] = _utc_now_str()
            try:
                _write_state_atomic(state)
            except Exception:
                pass

        for _ in range(max(1, POLL_INTERVAL_SEC)):
            if not _RUNNING:
                break
            time.sleep(1)

    state["status"] = "stopped"
    state["updated_utc"] = _utc_now_str()
    try:
        _write_state_atomic(state)
    except Exception:
        pass


if __name__ == "__main__":
    run_worker_loop()
