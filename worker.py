from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from mtf_runner import run_closed_candle_5tf_cascade
from notifications import NotificationCenter


DATA_DIR = Path("data")
SEEN_FILE = DATA_DIR / "seen_worker_alerts.json"

DEFAULT_WATCHLIST: dict[str, str] = {
    "Bitcoin": "BTC-USD",
    "Gold": "GC=F",
    "Crude Oil": "CL=F",
    "EUR/USD": "EURUSD=X",
    "GBP/JPY": "GBPJPY=X",
}

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _clean_df(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=OHLCV)

    df = frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]
    if not all(col in df.columns for col in required):
        return pd.DataFrame(columns=OHLCV)

    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df = df[OHLCV].copy()
    for col in OHLCV:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    valid = (
        (df["High"] >= df["Open"])
        & (df["High"] >= df["Close"])
        & (df["High"] >= df["Low"])
        & (df["Low"] <= df["Open"])
        & (df["Low"] <= df["Close"])
        & (df["Low"] <= df["High"])
    )
    return df.loc[valid]


def _fetch_yf(symbol: str, interval: str, period: str) -> pd.DataFrame:
    try:
        ticker = yf.Ticker(symbol)
        raw = ticker.history(
            period=period,
            interval=interval,
            auto_adjust=False,
            actions=False,
            prepost=False,
        )
        return _clean_df(raw)
    except Exception:
        return pd.DataFrame(columns=OHLCV)


def _resample_4h(hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly is None or hourly.empty:
        return pd.DataFrame(columns=OHLCV)
    try:
        resampled = hourly.resample("4h", label="right", closed="right").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })
        return _clean_df(resampled)
    except Exception:
        return pd.DataFrame(columns=OHLCV)


def load_5tf_frames(symbol: str) -> dict[str, pd.DataFrame]:
    hourly = _fetch_yf(symbol, "1h", "730d")
    return {
        "4H": _resample_4h(hourly),
        "1H": hourly,
        "15M": _fetch_yf(symbol, "15m", "60d"),
        "5M": _fetch_yf(symbol, "5m", "60d"),
        "1M": _fetch_yf(symbol, "1m", "7d"),
    }


def load_seen_ids() -> set[str]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SEEN_FILE.exists():
        return set()
    try:
        data = json.loads(SEEN_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {str(x) for x in data[-2000:]}
    except Exception:
        pass
    return set()


def save_seen_ids(seen_ids: set[str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    trimmed = list(seen_ids)[-2000:]
    SEEN_FILE.write_text(json.dumps(trimmed, indent=2), encoding="utf-8")


def run_once(
    notifier: NotificationCenter,
    seen_ids: set[str],
    rr_target: float = 5.0,
    tg_token: str = "",
    tg_chat_id: str = "",
) -> int:
    new_alerts_count = 0

    for asset_name, symbol in DEFAULT_WATCHLIST.items():
        frames = load_5tf_frames(symbol)
        df_1m = frames.get("1M", pd.DataFrame())
        if df_1m is None or len(df_1m) < 2:
            continue

        # Latest closed candle stamp (iloc[-2] is the last completed bar)
        closed_stamp = str(df_1m.index[-2])
        mtf_out = run_closed_candle_5tf_cascade(frames, rr_target=rr_target)
        sig = mtf_out.get("scalp_signal")

        if sig is not None:
            event_id = f"worker_scalp_{symbol}_{sig.direction}_{sig.entry_price}_{closed_stamp}"
            if event_id not in seen_ids:
                title = f"{asset_name} ({symbol}) {sig.direction} — {sig.confidence}"
                detail = (
                    f"Closed Bar: {closed_stamp} | "
                    f"Entry: {sig.entry_price:,.5f} | "
                    f"SL: {sig.stop_loss:,.5f} | "
                    f"TP: {sig.take_profit:,.5f} (1:{sig.rr_ratio:.1f} R:R) | "
                    f"HTF: {sig.htf_zone_summary} -> LTF: {sig.ltf_overlap_summary}"
                )
                pushed = notifier.push(
                    event_id=event_id,
                    category="SCALP_SIGNAL",
                    title=title,
                    detail=detail,
                    telegram_token=tg_token,
                    telegram_chat_id=tg_chat_id,
                )
                if pushed:
                    seen_ids.add(event_id)
                    new_alerts_count += 1
                    print(f"[{datetime.now(timezone.utc).isoformat()}] ALERT: {title} | {detail}")

    return new_alerts_count


def main() -> None:
    poll_seconds = int(os.getenv("POLL_SECONDS", "60"))
    rr_target = float(os.getenv("RR_TARGET", "5.0"))
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    tg_chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    run_once_mode = os.getenv("RUN_ONCE", "false").strip().lower() in ("1", "true", "yes")

    notifier = NotificationCenter()
    seen_ids = load_seen_ids()

    print(
        f"DESTINY AI Worker started | poll={poll_seconds}s | rr=1:{rr_target:.1f} | "
        f"telegram={'enabled' if (tg_token and tg_chat_id) else 'console-only'}"
    )

    while True:
        try:
            count = run_once(
                notifier=notifier,
                seen_ids=seen_ids,
                rr_target=rr_target,
                tg_token=tg_token,
                tg_chat_id=tg_chat_id,
            )
            if count > 0:
                save_seen_ids(seen_ids)
        except Exception as exc:
            print(f"[{datetime.now(timezone.utc).isoformat()}] Worker loop error: {exc}")

        if run_once_mode:
            break
        time.sleep(max(15, poll_seconds))


if __name__ == "__main__":
    main()

