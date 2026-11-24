
from __future__ import annotations

import inspect
import json
import logging
import os
import signal
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd
import yfinance as yf

from broker_and_account import VirtualAccount
from ob_fvg_engine import run_ob_fvg_engine
from sr_engine import run_sr_zones_engine

log = logging.getLogger("worker")

# FIX: a failed import is now logged loudly instead of silently disabling signals.
try:
    from strategy_engine import evaluate_setups as _evaluate_setups  # type: ignore
except Exception:
    log.exception("strategy_engine import failed - worker will produce NO signals")
    _evaluate_setups = None

try:
    import mtf_runner as _mtf_runner  # type: ignore
except Exception:
    log.warning("mtf_runner not available; continuing without MTF hooks", exc_info=True)
    _mtf_runner = None

STATE_PATH = Path("data/worker_state.json")
VA_DIR = Path("data/virtual_accounts")
POLL_INTERVAL_SEC = int(os.getenv("WORKER_POLL_SEC", "30"))
MAX_SIGNALS = int(os.getenv("WORKER_MAX_SIGNALS", "25"))
MAX_AUTO_OPEN_POSITIONS = int(os.getenv("MAX_AUTO_OPEN_POSITIONS", "3"))
# Slightly above 3 x 2% so rounding / balance drift does not block the third trade.
MAX_TOTAL_RISK_PCT = float(os.getenv("MAX_TOTAL_RISK_PCT", "6.5"))
MAX_RISK_PCT_PER_TRADE = float(os.getenv("MAX_RISK_PCT_PER_TRADE", "5.0"))
MAX_LEVERAGE = float(os.getenv("MAX_LEVERAGE", "20"))
MAX_CLOSED_LOG = int(os.getenv("MAX_CLOSED_LOG", "500"))
MARGIN_RATE = 0.05
BASE_TF_MIN = 15

WORKER_WATCHLIST: list[tuple[str, str]] = [
    ("Gold (XAU)", "GC=F"),
    ("Crude Oil (WTI)", "CL=F"),
    ("EUR/USD", "EURUSD=X"),
    ("GBP/USD", "GBPUSD=X"),
    ("Bitcoin (BTC)", "BTC-USD"),
]

# FIX: groups of instruments that are effectively the same bet (one USD view).
CORRELATED_GROUPS: list[set[str]] = [{"EURUSD=X", "GBPUSD=X"}]

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
_INTERVAL_MINUTES = {"15m": 15, "1h": 60}
_RUNNING = True
_WARNED_ANON = False


def _handle_stop(signum: int, frame: object) -> None:
    _ = (signum, frame)
    global _RUNNING
    _RUNNING = False


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_str() -> str:
    return _utc_now().strftime("%Y-%m-%d %H:%M:%S UTC")


def _utc_iso() -> str:
    return _utc_now().isoformat(timespec="seconds")


def _get_val(obj: object, names: tuple[str, ...], default: object = None) -> object:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _to_float(value: object, default: float | None = None) -> float | None:
    """FIX: safe conversion; "n/a", None, NaN and inf all become `default`."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if out != out or out in (float("inf"), float("-inf")):
        return default
    return out


def _num_or_str(value: object) -> object:
    """JSON-safe: a real float when possible, otherwise a string."""
    f = _to_float(value)
    return f if f is not None else str(value)


def _call_filtered(fn, **kwargs):
    """FIX: pass only the keyword args `fn` accepts, instead of guessing by
    catching TypeError (which also swallowed genuine bugs inside `fn`)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(**kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)
    return fn(**{k: v for k, v in kwargs.items() if k in params})


# --------------------------------------------------------------------------- #
# Position record (replaces the dynamic type(...) objects)
# --------------------------------------------------------------------------- #
class PositionRecord:
    """Plain position object that always carries the fields the worker needs.

    `reserved` is cash the account deducted when the position was opened
    (margin or risk, depending on VirtualAccount). It is refunded on close.
    """

    def __init__(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        size: float,
        units: float,
        margin_used: float,
        reserved: float = 0.0,
        opened_utc: str | None = None,
    ) -> None:
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.size = size
        self.units = units
        self.margin_used = margin_used
        self.reserved = reserved
        self.opened_utc = opened_utc

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "size": self.size,
            "units": self.units,
            "margin_used": self.margin_used,
            "reserved": self.reserved,
            "opened_utc": self.opened_utc,
        }


def _to_record(pos: object, reserved: float | None = None, opened_utc: str | None = None) -> PositionRecord:
    size = _to_float(_get_val(pos, ("size", "qty", "quantity", "units"), 1.0), 1.0)
    units = _to_float(_get_val(pos, ("units", "size", "qty", "quantity"), size), size)
    entry = _to_float(_get_val(pos, ("entry_price", "entry"), 0.0), 0.0)
    margin = _to_float(_get_val(pos, ("margin_used", "margin", "collateral"), None))
    if margin is None:
        margin = round(abs(units * entry) * MARGIN_RATE, 2)
    if reserved is None:
        reserved = _to_float(_get_val(pos, ("reserved",), 0.0), 0.0)
    return PositionRecord(
        symbol=str(_get_val(pos, ("symbol",), "")),
        side=str(_get_val(pos, ("side",), "LONG")).upper(),
        entry_price=entry,
        stop_loss=_to_float(_get_val(pos, ("stop_loss", "sl"), 0.0), 0.0),
        take_profit=_to_float(_get_val(pos, ("take_profit", "tp"), 0.0), 0.0),
        size=size,
        units=units,
        margin_used=margin,
        reserved=reserved,
        opened_utc=opened_utc or _get_val(pos, ("opened_utc",), None),
    )


def _record_from_dict(item: dict) -> PositionRecord:
    ent = _to_float(item.get("entry_price"), 0.0)
    sz = _to_float(item.get("size", item.get("units")), 1.0)
    units = _to_float(item.get("units"), sz)
    margin = _to_float(item.get("margin_used"))
    if margin is None:
        margin = round(abs(units * ent) * MARGIN_RATE, 2)
    return PositionRecord(
        symbol=str(item.get("symbol", "")),
        side=str(item.get("side", "LONG")).upper(),
        entry_price=ent,
        stop_loss=_to_float(item.get("stop_loss"), 0.0),
        take_profit=_to_float(item.get("take_profit"), 0.0),
        size=sz,
        units=units,
        margin_used=margin,
        reserved=_to_float(item.get("reserved"), 0.0),
        opened_utc=item.get("opened_utc"),
    )


# --------------------------------------------------------------------------- #
# Supabase (single request helper; failures are logged, not swallowed)
# --------------------------------------------------------------------------- #
def _supabase_config() -> tuple[str, str]:
    global _WARNED_ANON
    url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    service = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
    generic = os.getenv("SUPABASE_KEY", "").strip()
    anon = os.getenv("SUPABASE_ANON_KEY", "").strip()
    # FIX: prefer the service key for server-side writes (anon is often blocked by RLS).
    key = service or generic or anon
    if key and not (service or generic) and not _WARNED_ANON:
        log.warning(
            "Using SUPABASE_ANON_KEY for server-side writes; use SUPABASE_SERVICE_KEY "
            "(server only) or writes may be rejected by row-level security."
        )
        _WARNED_ANON = True
    return url, key


def _sb_request(method: str, path: str, body: object | None = None, prefer: str | None = None):
    """Returns (ok, parsed_json_or_None)."""
    url, key = _supabase_config()
    if not url or not key:
        return False, None
    headers = {"apikey": key, "Authorization": f"Bearer {key}", "Accept": "application/json"}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if prefer:
        headers["Prefer"] = prefer
    req = urllib.request.Request(f"{url}/rest/v1/{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read().decode("utf-8")
            parsed = json.loads(raw) if raw.strip() else None
            return 200 <= resp.status < 300, parsed
    except Exception as exc:
        log.warning("Supabase %s %s failed: %s", method, path.split("?")[0], exc)
        return False, None


def _extract_payload(rows: object) -> dict | None:
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        raw = rows[0].get("payload")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                return None
        if isinstance(raw, dict):
            return raw
    return None


def _supabase_load_worker_state() -> dict | None:
    ok, rows = _sb_request("GET", "worker_state?id=eq.default&select=payload&limit=1")
    return _extract_payload(rows) if ok else None


def _supabase_save_worker_state(state: dict) -> bool:
    ok, _ = _sb_request(
        "POST", "worker_state?on_conflict=id",
        [{"id": "default", "payload": state}], prefer="resolution=merge-duplicates",
    )
    return ok


def _supabase_load_account(username: str) -> dict | None:
    path = f"virtual_accounts?username=eq.{urllib.parse.quote(username)}&select=payload&limit=1"
    ok, rows = _sb_request("GET", path)
    return _extract_payload(rows) if ok else None


def _supabase_list_usernames() -> list[str]:
    ok, rows = _sb_request("GET", "virtual_accounts?select=username")
    if not ok or not isinstance(rows, list):
        return []
    return [
        str(r.get("username", "")).strip()
        for r in rows
        if isinstance(r, dict) and str(r.get("username", "")).strip()
    ]


def _supabase_save_account(username: str, payload: dict) -> bool:
    ok, _ = _sb_request(
        "POST", "virtual_accounts?on_conflict=username",
        [{"username": username, "payload": payload}], prefer="resolution=merge-duplicates",
    )
    return ok


# --------------------------------------------------------------------------- #
# Account storage
# --------------------------------------------------------------------------- #
def _va_file_path(username: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in username)[:40] or "default"
    VA_DIR.mkdir(parents=True, exist_ok=True)
    return VA_DIR / f"{safe_name}.json"


@contextmanager
def _account_lock(username: str, timeout: float = 10.0, stale_after: float = 60.0) -> Iterator[None]:
    """FIX: same-host lock around an account's load -> modify -> save cycle.

    app.py should wrap its own account writes in this too. It does not protect
    against two different machines writing to Supabase at once.
    """
    lock_path = _va_file_path(username).with_suffix(".lock")
    deadline = time.monotonic() + timeout
    fd = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > stale_after:
                    lock_path.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"could not lock account '{username}'")
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(fd)
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


def _load_raw_account_payload(username: str) -> dict:
    remote = _supabase_load_account(username)
    local: dict | None = None
    acct_path = _va_file_path(username)
    if acct_path.exists():
        try:
            data = json.loads(acct_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                local = data
        except Exception:
            log.warning("could not read local account file %s", acct_path, exc_info=True)

    # FIX: choose the newer copy instead of always trusting remote
    # (a failed remote save used to let a stale remote overwrite newer local data).
    if remote and local:
        return remote if str(remote.get("updated_utc", "")) >= str(local.get("updated_utc", "")) else local
    if remote:
        return remote
    if local:
        return local
    return {
        "balance": 10000.0,
        "positions": [],
        "closed_positions": [],
        "auto_exec_virtual": True,
        "risk_pct": 2.0,
    }


def _account_to_payload(acct: VirtualAccount, closed_log: list[dict], prefs: dict | None = None) -> dict:
    p = prefs or {}
    return {
        "balance": round(float(getattr(acct, "balance", 10000.0)), 2),
        "positions": [_to_record(pos).to_dict() for pos in list(getattr(acct, "positions", []) or [])],
        "closed_positions": list(closed_log or [])[-MAX_CLOSED_LOG:],
        "auto_exec_virtual": bool(p.get("auto_exec_virtual", True)),
        "risk_pct": float(p.get("risk_pct", 2.0)),
        "updated_utc": _utc_iso(),
    }


def _payload_to_account(data: dict) -> tuple[VirtualAccount, list[dict], dict]:
    acct = VirtualAccount()
    closed_log: list[dict] = []
    prefs = {"auto_exec_virtual": True, "risk_pct": 2.0}
    if not isinstance(data, dict):
        return acct, closed_log, prefs

    prefs["auto_exec_virtual"] = bool(data.get("auto_exec_virtual", True))
    prefs["risk_pct"] = _to_float(data.get("risk_pct"), 2.0)

    if "balance" in data and hasattr(acct, "balance"):
        bal = _to_float(data["balance"])
        if bal is not None:
            acct.balance = round(bal, 2)
    closed_log = list(data.get("closed_positions") or [])
    if hasattr(acct, "positions") and isinstance(acct.positions, list):
        for item in list(data.get("positions") or []):
            if not isinstance(item, dict):
                continue
            try:
                acct.positions.append(_record_from_dict(item))
            except Exception:
                log.exception("could not restore saved position: %s", item)
    return acct, closed_log, prefs


def _load_virtual_account_obj(username: str) -> tuple[VirtualAccount, list[dict], dict]:
    return _payload_to_account(_load_raw_account_payload(username))


def _save_virtual_account_obj(username: str, acct: VirtualAccount, closed_log: list[dict], prefs: dict | None = None) -> None:
    payload = _account_to_payload(acct, closed_log, prefs=prefs)
    try:
        path = _va_file_path(username)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)  # FIX: atomic write
    except Exception:
        log.exception("local account save failed for %s", username)
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
    users.update(_supabase_list_usernames())
    return sorted(users)


# --------------------------------------------------------------------------- #
# Opening positions
# --------------------------------------------------------------------------- #
def _levels_valid(side: str, entry: float, sl: float, tp: float) -> bool:
    """FIX: SL/TP must be on the correct side of entry for the direction."""
    if side == "LONG":
        return sl < entry < tp
    if side == "SHORT":
        return tp < entry < sl
    return False


def _open_risk_dollars(positions: list) -> float:
    total = 0.0
    for pos in positions:
        rec = _to_record(pos)
        total += abs(rec.entry_price - rec.stop_loss) * abs(rec.units)
    return total


def _exposure_block_reason(positions: list, symbol: str, side: str, new_risk: float, balance: float) -> str | None:
    """FIX: position cap + one position per symbol + correlation + total-risk cap."""
    if len(positions) >= MAX_AUTO_OPEN_POSITIONS:
        return "max_positions"
    for pos in positions:
        rec = _to_record(pos)
        if rec.symbol == symbol:
            return "symbol_already_open"
        for group in CORRELATED_GROUPS:
            if symbol in group and rec.symbol in group and rec.side == side:
                return "correlated_exposure"
    if balance > 0 and (_open_risk_dollars(positions) + new_risk) > balance * MAX_TOTAL_RISK_PCT / 100.0:
        return "total_risk_cap"
    return None


def _call_open_fn(open_fn, symbol, side, entry_px, sl_px, tp_px, risk_pct):
    """FIX: choose the call style from the signature instead of catching TypeError."""
    try:
        params = inspect.signature(open_fn).parameters
    except (TypeError, ValueError):
        params = {}
    has_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    if "risk_pct" in params or has_kwargs:
        return open_fn(symbol, side, entry_px, sl_px, tp_px, risk_pct=risk_pct)
    if len(params) >= 6:
        return open_fn(symbol, side, entry_px, sl_px, tp_px, risk_pct)
    return open_fn(symbol, side, entry_px, sl_px, tp_px)


def _stamp_new_positions(acct: VirtualAccount, before: int, reserved: float, opened_utc: str) -> None:
    """Convert newly opened positions to PositionRecord and stamp open time + reserved cash."""
    try:
        positions = acct.positions
        count = len(positions) - before
        if count <= 0:
            return
        per_position = round(reserved / count, 2)
        for i in range(before, len(positions)):
            positions[i] = _to_record(positions[i], reserved=per_position, opened_utc=opened_utc)
    except Exception:
        log.exception("could not stamp new position(s)")


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
    intended_risk = round(pre_balance * (float(risk_pct) / 100.0), 2)
    before = len(list(getattr(acct, "positions", []) or []))
    now_iso = _utc_iso()

    open_fn = getattr(acct, "open_position_with_risk", None)
    if callable(open_fn):
        try:
            res = _call_open_fn(open_fn, symbol, side, entry_px, sl_px, tp_px, risk_pct)
        except Exception:
            log.exception("open_position_with_risk raised for %s", symbol)
            return False, "open_fn_error"
        after = len(list(getattr(acct, "positions", []) or []))
        if isinstance(res, tuple) and len(res) >= 2:
            ok, msg = bool(res[0]), str(res[1])
        elif res is None:
            ok, msg = after > before, "Opened via VirtualAccount"
        else:
            ok, msg = bool(res), "Opened via VirtualAccount"
        if not ok:
            return False, msg
        # FIX: whatever cash the account deducted at open is remembered and refunded
        # on close. The old code forced balance = pre - risk and never gave it back.
        reserved = round(pre_balance - float(getattr(acct, "balance", pre_balance)), 2)
        _stamp_new_positions(acct, before, reserved, now_iso)
        return True, msg

    sl_dist = abs(entry_px - sl_px)
    if sl_dist <= 0 or intended_risk <= 0 or pre_balance <= 0:
        return False, "signal_only_risk_guard"
    units = intended_risk / sl_dist
    units = round(min(units, pre_balance * MAX_LEVERAGE / entry_px), 6)  # FIX: leverage cap
    if units <= 0:
        return False, "signal_only_risk_guard"
    rec = PositionRecord(
        symbol=symbol,
        side=side,
        entry_price=entry_px,
        stop_loss=sl_px,
        take_profit=tp_px,
        size=units,
        units=units,
        margin_used=round(abs(units * entry_px) * MARGIN_RATE, 2),
        reserved=0.0,  # FIX: nothing is deducted at open; P&L is applied at close
        opened_utc=now_iso,
    )
    positions = list(getattr(acct, "positions", []) or [])
    positions.append(rec)
    acct.positions = positions
    return True, "opened_fallback"


# --------------------------------------------------------------------------- #
# Closing positions
# --------------------------------------------------------------------------- #
def _bars_since(source: object, opened_utc: str | None) -> list[tuple[float, float, float]]:
    """(open, high, low) for bars that STARTED after the position opened.

    FIX: the old code tested the latest bar's full range, which can include
    prices from before the entry and close a trade instantly and falsely.
    """
    if isinstance(source, tuple):  # legacy input: (high, low, close)
        hi, lo, close = source
        return [(close, hi, lo)]
    df = source
    if df is None or getattr(df, "empty", True):
        return []
    opened_ts = pd.to_datetime(opened_utc, utc=True, errors="coerce") if opened_utc else pd.NaT
    if pd.isna(opened_ts):  # legacy position with no timestamp: latest bar only
        last = df.iloc[-1]
        return [(float(last["Open"]), float(last["High"]), float(last["Low"]))]
    idx = df.index
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    sub = df[idx > opened_ts]
    return [(float(r.Open), float(r.High), float(r.Low)) for r in sub.itertuples()]


def _check_exit(side: str, sl: float, tp: float, bars: list[tuple[float, float, float]]) -> float | None:
    """First bar (in time order) that touches SL or TP. SL wins if both touch in one bar."""
    for bar_open, hi, lo in bars:
        if side == "LONG":
            if sl > 0 and lo <= sl:
                return min(sl, bar_open)  # gap through the stop fills at the open
            if tp > 0 and hi >= tp:
                return tp
        elif side == "SHORT":
            if sl > 0 and hi >= sl:
                return max(sl, bar_open)
            if tp > 0 and lo <= tp:
                return tp
    return None


def _evaluate_tp_sl_hits_for_users(price_bars: dict, usernames: list[str]) -> int:
    """`price_bars` maps symbol -> OHLCV DataFrame (legacy (hi, lo, close) tuples still work)."""
    if not price_bars or not usernames:
        return 0

    total_closed = 0
    for username in usernames:
        try:
            with _account_lock(username):
                acct, closed_log, prefs = _load_virtual_account_obj(username)
                positions = list(getattr(acct, "positions", []) or [])
                if not positions:
                    continue

                remaining = []
                closed_count = 0
                for pos in positions:
                    rec = _to_record(pos)
                    source = price_bars.get(rec.symbol)
                    if source is None:
                        remaining.append(pos)
                        continue
                    exit_px = _check_exit(
                        rec.side, rec.stop_loss, rec.take_profit, _bars_since(source, rec.opened_utc)
                    )
                    if exit_px is None:
                        remaining.append(pos)
                        continue

                    signed_units = -abs(rec.units) if rec.side == "SHORT" else abs(rec.units)
                    pnl_val = round(float((exit_px - rec.entry_price) * signed_units), 2)
                    if hasattr(acct, "balance"):
                        # FIX: refund whatever was reserved at open, then apply P&L.
                        acct.balance = round(float(acct.balance) + rec.reserved + pnl_val, 2)

                    closed_log.append({
                        "symbol": rec.symbol,
                        "side": rec.side,
                        "entry": rec.entry_price,
                        "exit": float(exit_px),
                        "sl": rec.stop_loss,
                        "tp": rec.take_profit,
                        "pnl": pnl_val,
                        "closed_by": "worker_tp_sl",
                        "opened_utc": rec.opened_utc,
                        "closed_utc": _utc_iso(),
                    })
                    closed_count += 1

                if closed_count > 0:
                    acct.positions = remaining
                    _save_virtual_account_obj(username, acct, closed_log, prefs=prefs)
                    total_closed += closed_count
        except Exception:
            log.exception("TP/SL evaluation failed for %s", username)

    return total_closed


# --------------------------------------------------------------------------- #
# Auto-execution
# --------------------------------------------------------------------------- #
def _try_auto_execute_for_users(signal_item: dict, usernames: list[str], default_risk_pct: float = 2.0) -> str:
    if not usernames:
        return "auto_ready_no_trader"

    # FIX: "n/a" and other junk no longer raise ValueError and kill the whole cycle.
    entry_px = _to_float(signal_item.get("entry"))
    sl_px = _to_float(signal_item.get("sl"))
    tp_px = _to_float(signal_item.get("tp"))
    side = str(signal_item.get("direction", "")).upper()
    symbol = str(signal_item.get("symbol", "")).strip()
    if (
        entry_px is None or sl_px is None or tp_px is None
        or entry_px <= 0 or sl_px <= 0 or tp_px <= 0
        or side not in ("LONG", "SHORT") or not symbol
        or not _levels_valid(side, entry_px, sl_px, tp_px)
    ):
        return "signal_only_invalid_levels"

    executed_users: list[str] = []
    blocked: list[str] = []
    for username in usernames:
        try:
            with _account_lock(username):
                acct, closed_log, prefs = _load_virtual_account_obj(username)
                if not bool(prefs.get("auto_exec_virtual", True)):
                    continue
                positions = list(getattr(acct, "positions", []) or [])
                user_risk = _to_float(prefs.get("risk_pct"), default_risk_pct) or default_risk_pct
                user_risk = min(max(user_risk, 0.0), MAX_RISK_PCT_PER_TRADE)
                balance = float(getattr(acct, "balance", 0.0))
                reason = _exposure_block_reason(positions, symbol, side, balance * user_risk / 100.0, balance)
                if reason:
                    log.info("skip %s %s %s for %s: %s", side, symbol, entry_px, username, reason)
                    blocked.append(reason)
                    continue
                ok, msg = _open_via_virtual_account(acct, symbol, side, entry_px, sl_px, tp_px, risk_pct=user_risk)
                if ok:
                    _save_virtual_account_obj(username, acct, closed_log, prefs=prefs)
                    executed_users.append(username)
                else:
                    log.info("open rejected for %s on %s: %s", username, symbol, msg)
        except Exception:
            log.exception("auto-execute failed for %s", username)

    if executed_users:
        return f"virtual_executed ({', '.join(executed_users)})"
    if blocked:
        return f"auto_blocked ({', '.join(sorted(set(blocked)))})"
    return "auto_armed_no_new_fill"


# --------------------------------------------------------------------------- #
# Market data
# --------------------------------------------------------------------------- #
def _clean_ohlcv(df: pd.DataFrame | None, tail: int | None = 200) -> pd.DataFrame:
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
    out = out.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    return out.tail(tail) if tail else out


def _drop_forming_bar(df: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    """FIX: yfinance index = bar START time; the last bar is still forming until start + length."""
    if df is None or df.empty:
        return df
    last_ts = pd.Timestamp(df.index[-1])
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    if last_ts + pd.Timedelta(minutes=bar_minutes) > pd.Timestamp(_utc_now()):
        return df.iloc[:-1]
    return df


def _resample_4h(hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly is None or hourly.empty:
        return pd.DataFrame(columns=OHLCV)
    try:
        # FIX: label/closed="left". With closed="right" the 04:00 hour bar was
        # grouped into the 00:00-04:00 bucket (wrong bar membership).
        res = hourly.resample("4h", label="left", closed="left").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })
        return _clean_ohlcv(res)
    except Exception:
        log.exception("4h resample failed")
        return pd.DataFrame(columns=OHLCV)


def _download_tf(
    symbol: str,
    interval: str,
    period: str,
    resample_4h: bool = False,
    closed_only: bool = True,
) -> pd.DataFrame:
    """Download OHLCV. With closed_only=True the still-forming last bar is removed."""
    # FIX: do not truncate hourly data to 200 rows before resampling to 4H
    # (that left only ~50 4H bars despite requesting 120 days).
    keep = None if resample_4h else 200
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
        df = _clean_ohlcv(dl, tail=keep)
    except Exception as exc:
        log.warning("yf.download failed for %s %s: %s", symbol, interval, exc)

    if df.empty:
        try:
            raw = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False, actions=False)
            df = _clean_ohlcv(raw, tail=keep)
        except Exception as exc:
            log.warning("yf history fallback failed for %s %s: %s", symbol, interval, exc)

    if df.empty:
        return df

    bar_min = _INTERVAL_MINUTES.get(interval, 0)
    if closed_only and bar_min:
        df = _drop_forming_bar(df, bar_min)
    if resample_4h:
        df = _resample_4h(df)
        if closed_only:
            df = _drop_forming_bar(df, 240)
    return df


# --------------------------------------------------------------------------- #
# Worker state
# --------------------------------------------------------------------------- #
def _load_existing_state() -> dict:
    if STATE_PATH.exists():
        try:
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("signals", [])
                data.setdefault("seen_keys", [])
                data.setdefault("last_bar", {})
                return data
        except Exception:
            log.warning("could not read %s", STATE_PATH, exc_info=True)
    remote = _supabase_load_worker_state()
    if isinstance(remote, dict):
        remote.setdefault("signals", [])
        remote.setdefault("seen_keys", [])
        remote.setdefault("last_bar", {})
        return remote
    return {
        "status": "starting",
        "updated_utc": _utc_now_str(),
        "cycle_count": 0,
        "scanned_assets": [],
        "signals": [],
        "seen_keys": [],
        "last_bar": {},
    }


def _write_state_atomic(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = STATE_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp_path, STATE_PATH)
    _supabase_save_worker_state(state)


# --------------------------------------------------------------------------- #
# Strategy call + scan
# --------------------------------------------------------------------------- #
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
    """FIX: passes only the kwargs evaluate_setups accepts and lets real errors
    propagate (the caller logs them). The old TypeError fallback chain could
    silently drop the multi-timeframe inputs."""
    if _evaluate_setups is None or df_15m.empty:
        return []
    res = _call_filtered(
        _evaluate_setups,
        df=df_15m,
        symbol=symbol,
        tf_label="15M",
        sr_out=sr_out,
        ob_out=ob_out,
        rr_target=rr_target,
        df_1h=df_1h,
        df_4h=df_4h,
        mtf_context=mtf_context,
        mtf_data={"15M": df_15m, "1H": df_1h, "4H": df_4h, **mtf_context},
    )
    return list(res) if res is not None else []


def _scan_once(
    state: dict,
    risk_pct: float = 2.0,
    target_users: list[str] | None = None,
    force: bool = False,
) -> dict:
    signals: list[dict] = list(state.get("signals") or [])
    seen_keys: list[str] = list(state.get("seen_keys") or [])
    seen_set = set(seen_keys)
    last_bar: dict[str, str] = dict(state.get("last_bar") or {})
    scanned: list[str] = []
    warnings: list[str] = []
    price_bars: dict[str, pd.DataFrame] = {}

    users_to_process = target_users if target_users is not None else _discover_trader_usernames()

    if _evaluate_setups is None:
        warnings.append("strategy_engine unavailable - no signals are being produced")

    for asset_name, symbol in WORKER_WATCHLIST:
        # FIX: one failing symbol no longer aborts the whole scan.
        try:
            # Full frame (including the forming bar) is only used for TP/SL monitoring.
            df_full = _download_tf(symbol, interval="15m", period="14d", closed_only=False)
            if df_full.empty:
                warnings.append(f"{symbol}: no 15m data")
                continue
            price_bars[symbol] = df_full

            # FIX: signals are evaluated on CLOSED bars only.
            df_15m = _drop_forming_bar(df_full, BASE_TF_MIN)
            if df_15m.empty:
                continue
            bar_ts = str(df_15m.index[-1])
            scanned.append(f"{symbol}@{bar_ts}")

            if _evaluate_setups is None:
                continue
            # FIX: heavy work (1H/4H downloads, engines, strategy) runs once per closed bar.
            if not force and last_bar.get(symbol) == bar_ts:
                continue

            df_1h = _download_tf(symbol, interval="1h", period="60d")
            df_4h = _download_tf(symbol, interval="1h", period="120d", resample_4h=True)

            try:
                sr_out = run_sr_zones_engine(df_15m, tf_label="15M", invalidation_mode="Wick")
            except Exception as exc:
                log.exception("sr engine failed for %s", symbol)
                warnings.append(f"{symbol}: sr engine: {exc}")
                sr_out = {"active_zones": [], "minor_zones": []}

            try:
                ob_out = run_ob_fvg_engine(df_15m, timeframe_minutes=15)
            except Exception as exc:
                log.exception("ob/fvg engine failed for %s", symbol)
                warnings.append(f"{symbol}: ob/fvg engine: {exc}")
                ob_out = {}

            mtf_context: dict = {"15M": df_15m, "1H": df_1h, "4H": df_4h}
            if _mtf_runner is not None:
                for hook_name in ("build_mtf_context", "get_mtf_context", "enrich_chart_context"):
                    fn = getattr(_mtf_runner, hook_name, None)
                    if not callable(fn):
                        continue
                    try:
                        out = _call_filtered(
                            fn,
                            symbol=symbol,
                            tf_label="15M",
                            df=df_15m,
                            df_1h=df_1h,
                            df_4h=df_4h,
                            sr_out=sr_out,
                            ob_out=ob_out,
                        )
                    except Exception as exc:
                        log.exception("mtf hook %s failed for %s", hook_name, symbol)
                        warnings.append(f"{symbol}: mtf hook {hook_name}: {exc}")
                        break
                    if isinstance(out, dict):
                        sr_out = out.get("sr_out", sr_out)
                        ob_out = out.get("ob_out", ob_out)
                        mtf_context.update(out)
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

            for s in setups:
                direction = str(_get_val(s, ("direction",), "SIGNAL"))
                conf = str(_get_val(s, ("confidence",), "15M/1H/4H"))
                entry = _num_or_str(_get_val(s, ("entry",), "n/a"))
                sl = _num_or_str(_get_val(s, ("sl",), "n/a"))
                tp = _num_or_str(_get_val(s, ("tp",), "n/a"))
                rr = _num_or_str(_get_val(s, ("rr",), 5.0))

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

            last_bar[symbol] = bar_ts  # only after a successful evaluation
        except Exception as exc:
            log.exception("scan failed for %s", symbol)
            warnings.append(f"{symbol}: {exc}")

    closed_hits = _evaluate_tp_sl_hits_for_users(price_bars, usernames=users_to_process)
    state["last_tp_sl_closed"] = closed_hits
    state["scanned_assets"] = scanned
    state["signals"] = signals[-MAX_SIGNALS:]
    state["seen_keys"] = seen_keys[-200:]
    state["last_bar"] = last_bar
    state["warnings"] = warnings[-20:]
    return state


def run_worker_scan_once(
    status_label: str = "auto_scan_ok",
    risk_pct: float = 2.0,
    target_users: list[str] | None = None,
    force: bool = False,
) -> dict:
    """`force=True` re-evaluates the latest closed bar even if it was already scanned."""
    state = _load_existing_state()
    try:
        state["cycle_count"] = int(state.get("cycle_count", 0)) + 1
        state = _scan_once(state, risk_pct=risk_pct, target_users=target_users, force=force)
        state["status"] = status_label
        state["updated_utc"] = _utc_now_str()
        _write_state_atomic(state)
    except Exception as exc:
        log.exception("scan cycle failed")
        state["status"] = f"error: {exc}"
        state["updated_utc"] = _utc_now_str()
        try:
            _write_state_atomic(state)
        except Exception:
            log.exception("could not persist error state")
    return state


def run_worker_loop() -> None:
    # FIX: signal handlers can only be installed from the main thread.
    try:
        signal.signal(signal.SIGINT, _handle_stop)
        signal.signal(signal.SIGTERM, _handle_stop)
    except ValueError:
        log.warning("not in main thread; signal handlers not installed")

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
            log.exception("scan cycle failed")
            state["status"] = f"error: {exc}"
            state["updated_utc"] = _utc_now_str()
            try:
                _write_state_atomic(state)
            except Exception:
                log.exception("could not persist error state")

        for _ in range(max(1, POLL_INTERVAL_SEC)):
            if not _RUNNING:
                break
            time.sleep(1)

    state["status"] = "stopped"
    state["updated_utc"] = _utc_now_str()
    try:
        _write_state_atomic(state)
    except Exception:
        log.exception("could not persist final state")


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # FIX: refuse to run as a silent no-op if the strategy could not be imported.
    if _evaluate_setups is None and os.getenv("ALLOW_NO_STRATEGY", "0") != "1":
        log.critical("strategy_engine failed to import; refusing to start (set ALLOW_NO_STRATEGY=1 to override)")
        raise SystemExit(1)
    run_worker_loop()
