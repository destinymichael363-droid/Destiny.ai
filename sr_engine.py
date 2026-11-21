
# =========================================================
# DESTINY AI - SUPPORT & RESISTANCE ENGINE
# CLOSE-BASED INVALIDATION
# =========================================================

import pandas as pd
import numpy as np

MAX_SR_CANDLES = 400


def calculate_atr(data, period=14):
    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    previous_close = close.shift(1)

    true_range = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs()
    ], axis=1).max(axis=1)

    return true_range.rolling(period, min_periods=1).mean()


def find_confirmed_pivots(data, length=8):
    pivots = []

    if len(data) < (length * 2 + 1):
        return pivots

    highs = data["High"].to_numpy()
    lows = data["Low"].to_numpy()

    for i in range(length, len(data) - length):

        window_high = highs[i - length:i + length + 1]
        window_low = lows[i - length:i + length + 1]

        if highs[i] == np.max(window_high):
            pivots.append({
                "index": i,
                "price": highs[i],
                "side": "Resistance"
            })

        if lows[i] == np.min(window_low):
            pivots.append({
                "index": i,
                "price": lows[i],
                "side": "Support"
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

    data = data.copy()

    required = ["High", "Low", "Close"]

    if not all(column in data.columns for column in required):
        return pd.DataFrame()

    atr = calculate_atr(data, atr_length)

    pivots = find_confirmed_pivots(data, pivot_length)

    zones = []

    for pivot in pivots:

        index = pivot["index"]
        price = pivot["price"]
        side = pivot["side"]

        current_atr = atr.iloc[index]

        if pd.isna(current_atr) or current_atr <= 0:
            continue

        width = current_atr * zone_atr

        if side == "Support":
            zone_low = price - width
            zone_high = price
        else:
            zone_low = price
            zone_high = price + width

        # -------------------------------------------------
        # CLOSE-BASED INVALIDATION
        # Support: close below zone
        # Resistance: close above zone
        # -------------------------------------------------

        invalidated = False

        for j in range(index + 1, len(data)):

            candle_close = data["Close"].iloc[j]

            if side == "Support" and candle_close < zone_low:
                invalidated = True
                break

            if side == "Resistance" and candle_close > zone_high:
                invalidated = True
                break

        # Never return invalidated zones to the chart
        if invalidated:
            continue

        # -------------------------------------------------
        # REJECTION / REACTION COUNT
        # -------------------------------------------------

        reactions = 0
        departed = True

        for j in range(index + 1, len(data)):

            candle_high = data["High"].iloc[j]
            candle_low = data["Low"].iloc[j]
            candle_close = data["Close"].iloc[j]

            if side == "Support":

                if departed and candle_low <= zone_high:
                    if candle_close > zone_high:
                        reactions += 1
                        departed = False

                if candle_close > zone_high + (current_atr * reaction_atr):
                    departed = True

            else:

                if departed and candle_high >= zone_low:
                    if candle_close < zone_low:
                        reactions += 1
                        departed = False

                if candle_close < zone_low - (current_atr * reaction_atr):
                    departed = True

        zones.append({
            "Side": side,
            "Zone Low": zone_low,
            "Zone High": zone_high,
            "Confirmed Time": data.index[index],
            "Rejections": reactions,
            "Strength": min(reactions, 3),
            "Midline": (zone_low + zone_high) / 2
        })

    if not zones:
        return pd.DataFrame()

    result = pd.DataFrame(zones)

    # Keep the newest active zones, not broken levels
    result = result.sort_values("Confirmed Time", ascending=False)

    return result.head(40).reset_index(drop=True)
