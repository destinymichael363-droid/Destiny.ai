
# =========================================================
# DESTINY AI - FAST SUPPORT & RESISTANCE ENGINE
# Vectorized historical zone analysis
# =========================================================

import pandas as pd
import numpy as np

MAX_SR_CANDLES = 400


def calculate_atr(data, length=14):

    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    previous = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous).abs(),
            (low - previous).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        span=length,
        adjust=False
    ).mean()


def find_confirmed_pivots(data, length=8):

    highs = data["High"].to_numpy(dtype=float)
    lows = data["Low"].to_numpy(dtype=float)

    pivots = []

    for i in range(length, len(data) - length):

        left_high = highs[i-length:i].max()
        right_high = highs[i+1:i+length+1].max()

        left_low = lows[i-length:i].min()
        right_low = lows[i+1:i+length+1].min()

        if highs[i] >= left_high and highs[i] > right_high:

            pivots.append((
                "RESISTANCE",
                i,
                i + length,
                highs[i]
            ))

        if lows[i] <= left_low and lows[i] < right_low:

            pivots.append((
                "SUPPORT",
                i,
                i + length,
                lows[i]
            ))

    return pivots


def analyze_sr(
    data,
    pivot_length=8,
    zone_atr=0.30,
    reaction_atr=0.50,
    atr_length=14
):

    if data is None or data.empty:
        return pd.DataFrame()

    required = ["Open", "High", "Low", "Close"]

    if not all(c in data.columns for c in required):
        return pd.DataFrame()

    df = data[required].copy()

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    ).dropna()

    # Only the latest 400 candles are processed.
    df = df.tail(MAX_SR_CANDLES).copy()

    if len(df) < pivot_length * 2 + atr_length:
        return pd.DataFrame()

    df["ATR"] = calculate_atr(df, atr_length)

    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    atr = df["ATR"].to_numpy(dtype=float)

    times = df.index

    pivots = find_confirmed_pivots(
        df,
        pivot_length
    )

    records = []

    for side, pivot_i, confirmed_i, price in pivots:

        atr_at_pivot = atr[pivot_i]

        if not np.isfinite(atr_at_pivot) or atr_at_pivot <= 0:
            continue

        width = atr_at_pivot * zone_atr

        lower = price - width
        upper = price + width

        start = confirmed_i + 1

        future_high = high[start:]
        future_low = low[start:]
        future_close = close[start:]
        future_atr = atr[start:]

        valid = (
            np.isfinite(future_atr)
            & (future_atr > 0)
        )

        touched = (
            (future_high >= lower)
            & (future_low <= upper)
        )

        if side == "SUPPORT":

            broken = future_close < lower

            reacted = (
                touched
                & (future_close > upper)
                & (
                    future_close - upper
                    >= future_atr * reaction_atr
                )
            )

        else:

            broken = future_close > upper

            reacted = (
                touched
                & (future_close < lower)
                & (
                    lower - future_close
                    >= future_atr * reaction_atr
                )
            )

        touches = int(np.count_nonzero(touched & valid))
        reactions = int(np.count_nonzero(reacted & valid))
        breaks = int(np.count_nonzero(broken & valid))

        if breaks > 0:
            status = "BROKEN"
            strength = "Invalidated"

        elif touches == 0:
            status = "FRESH"
            strength = "Unconfirmed"

        elif reactions >= 3:
            status = "REPEATED REACTIONS"
            strength = "High historical reaction"

        elif reactions >= 1:
            status = "REACTED"
            strength = "Moderate historical reaction"

        else:
            status = "TESTED"
            strength = "Unconfirmed"

        records.append({
            "Side": side,
            "Price": round(float(price), 8),
            "Zone Low": round(float(lower), 8),
            "Zone High": round(float(upper), 8),
            "Pivot Time": times[pivot_i],
            "Confirmed Time": times[confirmed_i],
            "Touches": touches,
            "Reactions": reactions,
            "Breaks": breaks,
            "Status": status,
            "Strength": strength
        })

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)

    return result.sort_values(
        "Confirmed Time",
        ascending=False
    ).reset_index(drop=True)
