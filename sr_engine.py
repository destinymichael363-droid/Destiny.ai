# =========================================================
# DESTINY AI - SUPPORT & RESISTANCE ENGINE
# Historical swing-based zone analysis
# =========================================================

import pandas as pd


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

    highs = data["High"].values
    lows = data["Low"].values

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

        # Pivot is only considered confirmed
        # after the right-side candles exist.
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

    required = [
        "Open",
        "High",
        "Low",
        "Close"
    ]

    if not all(
        column in data.columns
        for column in required
    ):

        return pd.DataFrame()

    df = data.copy()

    df = df.sort_index()

    df = df[
        ~df.index.duplicated(keep="last")
    ]

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=required
    )

    if len(df) < (pivot_length * 2 + atr_length):

        return pd.DataFrame()

    df["ATR"] = calculate_atr(
        df,
        atr_length
    )

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

        atr_value = df["ATR"].iloc[pivot_index]

        if pd.isna(atr_value) or atr_value <= 0:

            continue

        width = float(atr_value) * zone_atr

        lower = price - width
        upper = price + width

        touches = 0
        reactions = 0
        breaks = 0

        reaction_seen = False

        # =================================================
        # COUNT ONLY EVENTS AFTER PIVOT CONFIRMATION
        # =================================================

        future = df.iloc[
            confirmed_index + 1:
        ]

        for _, candle in future.iterrows():

            candle_high = float(candle["High"])
            candle_low = float(candle["Low"])
            candle_close = float(candle["Close"])

            atr_now = candle["ATR"]

            if pd.isna(atr_now) or atr_now <= 0:

                continue

            touched = (
                candle_high >= lower
                and candle_low <= upper
            )

            if touched:

                touches += 1

                reaction_seen = False

            # ---------------------------------------------
            # SUPPORT
            # ---------------------------------------------

            if pivot["side"] == "SUPPORT":

                if candle_close < lower:

                    breaks += 1

                elif (
                    touched
                    and candle_close > upper
                    and candle_close - upper
                    >= atr_now * reaction_atr
                ):

                    reactions += 1

            # ---------------------------------------------
            # RESISTANCE
            # ---------------------------------------------

            elif pivot["side"] == "RESISTANCE":

                if candle_close > upper:

                    breaks += 1

                elif (
                    touched
                    and candle_close < lower
                    and lower - candle_close
                    >= atr_now * reaction_atr
                ):

                    reactions += 1

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

    # Newest confirmed zones first
    result = result.sort_values(
        "Confirmed Time",
        ascending=False
    ).reset_index(drop=True)

    return result
