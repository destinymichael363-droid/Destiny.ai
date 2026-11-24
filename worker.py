from __future__ import annotations

import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from ob_fvg_engine import run_ob_fvg_engine
from sr_engine import run_sr_zones_engine

try:
    from strategy_engine import evaluate_setups as _evaluate_setups  # type: ignore
except Exception:
    _evaluate_setups = None

STATE_PATH = Path("data/worker_state.json")
POLL_INTERVAL_SEC = int(os.getenv("WORKER_POLL_SEC", "30"))
MAX_SIGNALS = int(os.getenv("WORKER_MAX_SIGNALS", "25"))

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


def _fetch_15m(symbol: str) -> pd.DataFrame:
    try:
        dl = yf.download(
            tickers=symbol,
            period="14d",
            interval="15m",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )
        cleaned = _clean_ohlcv(dl)
        if not cleaned.empty:
            return cleaned
    except Exception:
        pass
    try:
        raw = yf.Ticker(symbol).history(period="14d", interval="15m", auto_adjust=False, actions=False)
        return _clean_ohlcv(raw)
    except Exception:
        return pd.DataFrame(columns=OHLCV)


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
    return {
        "status": "starting",
        "updated_utc": _utc_now_str(),
        "cycle_count": 0,
        "signals": [],
        "seen_keys": [],
    }


def _write_state_atomic(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = STATE_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp_path, STATE_PATH)


def _scan_once(state: dict) -> dict:
    signals: list[dict] = list(state.get("signals") or [])
    seen_keys: list[str] = list(state.get("seen_keys") or [])
    seen_set = set(seen_keys)

    if _evaluate_setups is None:
        return state

    for asset_name, symbol in WORKER_WATCHLIST:
        df = _fetch_15m(symbol)
        if df.empty:
            continue

        try:
            sr_out = run_sr_zones_engine(df, tf_label="15M", invalidation_mode="Wick")
        except Exception:
            sr_out = {"active_zones": [], "minor_zones": []}

        try:
            ob_out = run_ob_fvg_engine(df, timeframe_minutes=15)
        except Exception:
            ob_out = {}

        try:
            setups = _evaluate_setups(
                df=df,
                symbol=symbol,
                tf_label="15M",
                sr_out=sr_out,
                ob_out=ob_out,
                rr_target=5.0,
            ) or []
        except Exception:
            setups = []

        bar_ts = str(df.index[-1])
        for s in setups:
            direction = str(s.get("direction") if isinstance(s, dict) else getattr(s, "direction", "SIGNAL"))
            conf = str(s.get("confidence") if isinstance(s, dict) else getattr(s, "confidence", "15M"))
            entry = s.get("entry") if isinstance(s, dict) else getattr(s, "entry", "n/a")
            sl = s.get("sl") if isinstance(s, dict) else getattr(s, "sl", "n/a")
            tp = s.get("tp") if isinstance(s, dict) else getattr(s, "tp", "n/a")
            rr = s.get("rr") if isinstance(s, dict) else getattr(s, "rr", 5.0)

            dedupe_key = f"{symbol}|15M|{direction}|{bar_ts}|{entry}"
            if dedupe_key in seen_set:
                continue
            seen_set.add(dedupe_key)
            seen_keys.append(dedupe_key)

            signals.append({
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
                "execution_status": "signal_only",
            })

    state["signals"] = signals[-MAX_SIGNALS:]
    state["seen_keys"] = seen_keys[-200:]
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
