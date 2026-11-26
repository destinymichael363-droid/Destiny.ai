"""
data_loader.py — Clean, cached OHLCV fetcher for DESTINY AI.

Shared by app.py (chart rendering) and worker.py (scanning).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from config import OHLCV, INTERVAL_MINUTES


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clean_ohlcv(
    df: pd.DataFrame | None,
    tail: int | None = None,
) -> pd.DataFrame:
    """
    Normalise a raw yfinance DataFrame to the standard OHLCV format.

    Returns an empty DataFrame with OHLCV columns if the input is
    unusable.
    """
    empty = pd.DataFrame(columns=OHLCV)

    if df is None or df.empty:
        return empty

    out = df.copy()

    # yfinance sometimes returns a MultiIndex for single-ticker downloads.
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)

    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(out.columns):
        return empty

    if "Volume" not in out.columns:
        out["Volume"] = 0.0

    out = out[OHLCV].copy()

    for col in OHLCV:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()

    if tail is not None and tail > 0:
        out = out.tail(tail)

    return out


def _drop_forming_bar(
    df: pd.DataFrame,
    bar_minutes: int,
) -> pd.DataFrame:
    """
    Remove the current still-forming candle (yfinance timestamps are
    bar START times).
    """
    if df is None or df.empty:
        return df

    last_ts = pd.Timestamp(df.index[-1])

    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")

    if last_ts + pd.Timedelta(minutes=bar_minutes) > pd.Timestamp(_utc_now()):
        return df.iloc[:-1]

    return df


def _resample_4h(hourly: pd.DataFrame) -> pd.DataFrame:
    """Resample 1H OHLCV into 4H candles."""
    if hourly is None or hourly.empty:
        return pd.DataFrame(columns=OHLCV)

    try:
        res = hourly.resample(
            "4h",
            label="left",
            closed="left",
        ).agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        return _clean_ohlcv(res)
    except Exception:
        return pd.DataFrame(columns=OHLCV)


def fetch_ohlcv(
    symbol: str,
    interval: str = "15m",
    period: str = "14d",
    *,
    closed_only: bool = True,
    tail: int | None = None,
    resample_4h: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV for one symbol and one timeframe.

    Parameters
    ----------
    symbol : yfinance ticker (e.g. "GC=F", "BTC-USD", "EURUSD=X")
    interval : "1m" | "5m" | "15m" | "1h"
    period : yfinance period string (e.g. "7d", "60d", "120d")
    closed_only : drop the current still-forming candle
    tail : keep only the last N candles (None = all)
    resample_4h : build 4H from 1H source instead of downloading 4H directly

    Returns a DataFrame with columns: Open, High, Low, Close, Volume.
    Empty DataFrame if the fetch failed.
    """
    if interval not in INTERVAL_MINUTES:
        raise ValueError(f"unsupported interval: {interval!r}")

    # Don't truncate hourly source data before 4H resampling.
    if resample_4h:
        keep = None
    elif tail is not None:
        keep = tail
    else:
        keep = 200

    df = pd.DataFrame(columns=OHLCV)

    # First attempt: bulk download.
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
    except Exception:
        pass

    # Fallback: Ticker.history() — sometimes works when download fails.
    if df.empty:
        try:
            raw = yf.Ticker(symbol).history(
                period=period,
                interval=interval,
                auto_adjust=False,
                actions=False,
            )
            df = _clean_ohlcv(raw, tail=keep)
        except Exception:
            pass

    if df.empty:
        return df

    bar_min = INTERVAL_MINUTES.get(interval, 0)

    if closed_only and bar_min:
        df = _drop_forming_bar(df, bar_min)

    if resample_4h:
        df = _resample_4h(df)
        if closed_only and not df.empty:
            df = _drop_forming_bar(df, 240)

    return df
