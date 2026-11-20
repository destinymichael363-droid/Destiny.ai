
# =========================================================
# DESTINY AI - OPTIMIZED SUPPORT & RESISTANCE ENGINE
# Historical swing-based zone analysis
# =========================================================

import pandas as pd
import numpy as np


# Maximum historical candles processed by the S/R engine.
# The original market data remains unchanged.
MAX_SR_CANDLES = 1200


def calculate_atr(data, length=14):

    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        span=length,
        adjust=False
    ).mean()


def find_confirmed_pivots(data, length=8):

    highs = data["High"].to_numpy()
    lows = data["Low"].to_numpy()

    pivots = []

    for i in range(length, len(data) - length):

        left_highs = highs[i-length:i]
        right_highs = highs[i+1:i+length+1]

        left_lows = lows[i-length:i]
        right_lows = lows[i+1:i+length+1]

        is_high = (
            highs[i] >= left_highs.max()
            and highs[i] > right_highs.max()
        )

        is_low = (
            lows[i] <= left_lows.min()
            and lows[i] < right_lows.min()
        )

        confirmed_at = i + length

        if is_high:

            pivots.append({
                "side": "RESISTANCE",
                "pivot_index": i,
                "confirmed_index": confirmed_at,
                "price": float(highs[i])
            })

        if is_low:

            pivots.append({
                "side": "SUPPORT",
                "pivot_index": i,
                "confirmed_index": confirmed_at,
                "price": float(lows[i])
            })

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

    if not all(column in data.columns for column in required):
        return pd.DataFrame()

    df = data.copy()

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(subset=required)

    # Limit the S/R calculation workload.
    df = df.tail(MAX_SR_CANDLES).copy()

    if len(df) < pivot_length * 2 + atr_length:
        return pd.DataFrame()

    df["ATR"] = calculate_atr(df, atr_length)

    # Convert columns to NumPy arrays for faster calculations.
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    atr_values = df["ATR"].to_numpy(dtype=float)

    pivots = find_confirmed_pivots(
        df,
        pivot_length
    )

    zones = []

    # =====================================================
    # BUILD SWING ZONES
    # =====================================================

    for pivot in pivots:

        pivot_index = pivot["pivot_index"]
        confirmed_index = pivot["confirmed_index"]

        price = pivot["price"]

        atr_value = atr_values[pivot_index]

        if not np.isfinite(atr_value) or atr_value <= 0:
            continue

        width = float(atr_value) * zone_atr

        lower = price - width
        upper = price + width

        # Only candles after pivot confirmation are evaluated.
        start = confirmed_index + 1

        future_highs = highs[start:]
        future_lows = lows[start:]
        future_closes = closes[start:]
        future_atr = atr_values[start:]

        if len(future_closes) == 0:
            touches = 0
            reactions = 0
            breaks = 0

        else:

            valid = (
                np.isfinite(future_atr)
                & (future_atr > 0)
            )

            touched = (
                (future_highs >= lower)
                & (future_lows <= upper)
            )

            touches = int(
                np.count_nonzero(touched & valid)
            )

            if pivot["side"] == "SUPPORT":

                broken = future_closes < lower

                reacted = (
                    touched
                    & (future_closes > upper)
                    & (
                        future_closes - upper
                        >= future_atr * reaction_atr
                    )
                )

            else:

                broken = future_closes > upper

                reacted = (
                    touched
                    & (future_closes < lower)
                    & (
                        lower - future_closes
                        >= future_atr * reaction_atr
                    )
                )

            breaks = int(
                np.count_nonzero(broken & valid)
            )

            reactions = int(
                np.count_nonzero(reacted & valid)
            )

        # =================================================
        # ZONE CLASSIFICATION
        # =================================================

        if breaks > 0:
            status = "BROKEN"

        elif touches == 0:
            status = "FRESH"

        elif reactions >= 3:
            status = "REPEATED REACTIONS"

        elif reactions >= 1:
            status = "REACTED"

        else:
            status = "TESTED"

        # =================================================
        # STRENGTH CLASSIFICATION
        # =================================================

        if breaks > 0:
            strength = "Invalidated"

        elif reactions >= 3:
            strength = "High historical reaction"

        elif reactions >= 1:
            strength = "Moderate historical reaction"

        else:
            strength = "Unconfirmed"

        zones.append({

            "Side": pivot["side"],

            "Price": round(price, 8),

            "Zone Low": round(lower, 8),

            "Zone High": round(upper, 8),

            "Pivot Time": df.index[pivot_index],

            "Confirmed Time": df.index[confirmed_index],

            "Touches": touches,

            "Reactions": reactions,

            "Breaks": breaks,

            "Status": status,

            "Strength": strength

        })

    if not zones:
        return pd.DataFrame()

    result = pd.DataFrame(zones)

    result = result.sort_values(
        "Confirmed Time",
        ascending=False
    ).reset_index(drop=True)

    return result
