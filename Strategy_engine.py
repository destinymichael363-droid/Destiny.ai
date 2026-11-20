
"""
DESTINY AI — BOS + INSTITUTIONAL OB + FVG + BSL/SSL
Educational historical market-analysis engine.
Input: OHLCV pandas DataFrame.
"""

import pandas as pd


DEFAULT_SETTINGS = {
    "pivot_length": 5,
    "ob_lookback": 20,
    "min_ob_atr": 0.35,
    "displacement_search": 8,
    "min_displacement_atr": 0.80,
    "min_close_location": 0.65,
    "fvg_search": 24,
    "atr_length": 14,
    "min_fvg_atr": 0.20,
    "use_volume_filter": True,
    "volume_length": 20,
    "min_fvg_volume_ratio": 1.5,
    "max_liquidity_levels": 30,
    "max_historical_setups": 100,
}


def calculate_atr(df, length=14):
    previous_close = df["Close"].shift(1)

    ranges = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - previous_close).abs(),
        (df["Low"] - previous_close).abs()
    ], axis=1)

    true_range = ranges.max(axis=1)

    return true_range.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length
    ).mean()


def analyze_market(data, settings=None):
    """
    Detect BOS, qualified order blocks, overlapping FVGs,
    BSL/SSL liquidity and historical setup invalidation.

    Returns:
        candles: OHLCV plus indicator columns
        setups: historical qualified OB + FVG setups
        liquidity: BSL/SSL levels and sweep information
        statistics: setup counts
    """

    cfg = DEFAULT_SETTINGS.copy()

    if settings:
        cfg.update(settings)

    df = data.copy()

    required = ["Open", "High", "Low", "Close", "Volume"]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    df = df[required].copy()

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if len(df) < max(cfg["atr_length"] + 5,
                     cfg["pivot_length"] * 2 + 5):
        raise ValueError("Not enough historical candles for analysis.")

    df["ATR"] = calculate_atr(df, cfg["atr_length"])
    df["AverageVolume"] = df["Volume"].rolling(
        cfg["volume_length"]
    ).mean()

    df["Bullish_BOS"] = False
    df["Bearish_BOS"] = False
    df["Bullish_Setup"] = False
    df["Bearish_Setup"] = False
    df["BSL_Sweep"] = False
    df["SSL_Sweep"] = False

    last_high = None
    last_low = None
    last_high_broken = False
    last_low_broken = False

    liquidity_levels = []
    all_liquidity = []
    active_zones = []
    setups = []

    pending_bull = None
    pending_bear = None

    invalidated_count = 0
    total_bullish_bos = 0
    total_bearish_bos = 0

    pivot_length = cfg["pivot_length"]

    def find_pivot(position, side):
        start = position - pivot_length
        end = position + pivot_length + 1

        if start < 0 or end > len(df):
            return False

        values = (
            df["High"].iloc[start:end]
            if side == "high"
            else df["Low"].iloc[start:end]
        )

        center = values.iloc[pivot_length]

        if side == "high":
            return center == values.max() and (values == center).sum() == 1

        return center == values.min() and (values == center).sum() == 1

    def check_fvg(position, direction, ob_high, ob_low):
        if position < 2:
            return None

        row = df.iloc[position]
        two_back = df.iloc[position - 2]

        atr = row["ATR"]

        if pd.isna(atr) or atr <= 0:
            return None

        volume_pass = True

        if cfg["use_volume_filter"]:
            previous = df.iloc[position - 1]

            volume_pass = (
                pd.notna(previous["Volume"])
                and pd.notna(previous["AverageVolume"])
                and previous["AverageVolume"] > 0
                and previous["Volume"]
                >= previous["AverageVolume"]
                * cfg["min_fvg_volume_ratio"]
            )

        if not volume_pass:
            return None

        if direction == 1:
            gap_low = two_back["High"]
            gap_high = row["Low"]

            valid_gap = (
                gap_high > gap_low
                and gap_high - gap_low >= atr * cfg["min_fvg_atr"]
            )

        else:
            gap_high = two_back["Low"]
            gap_low = row["High"]

            valid_gap = (
                gap_high > gap_low
                and gap_high - gap_low >= atr * cfg["min_fvg_atr"]
            )

        if not valid_gap:
            return None

        overlap = (
            max(ob_low, gap_low)
            < min(ob_high, gap_high)
        )

        if not overlap:
            return None

        return {
            "fvg_high": float(gap_high),
            "fvg_low": float(gap_low),
            "fvg_time": df.index[position]
        }

    def find_setup(position, direction, ob):
        start = max(ob["position"] + 1,
                    position - cfg["fvg_search"])

        for fvg_position in range(position, start - 1, -1):
            result = check_fvg(
                fvg_position,
                direction,
                ob["ob_high"],
                ob["ob_low"]
            )

            if result:
                return result

        return None

    def create_setup(direction, ob, fvg, bos_time):
        setup = {
            "direction": "BULLISH" if direction == 1 else "BEARISH",
            "bos_time": bos_time,
            "ob_time": ob["ob_time"],
            "ob_high": ob["ob_high"],
            "ob_low": ob["ob_low"],
            "fvg_time": fvg["fvg_time"],
            "fvg_high": fvg["fvg_high"],
            "fvg_low": fvg["fvg_low"],
            "status": "ACTIVE"
        }

        setups.append(setup)
        active_zones.append(setup)

        return setup

    for i in range(len(df)):

        row = df.iloc[i]
        current_time = df.index[i]

        # 1. Confirm swing pivots and identify liquidity.

        pivot_position = i - pivot_length

        if pivot_position >= pivot_length:

            if find_pivot(pivot_position, "high"):
                price = float(df["High"].iloc[pivot_position])

                last_high = price
                last_high_broken = False

                if not any(
                    level["side"] == "BSL"
                    and level["price"] == price
                    for level in liquidity_levels
                ):
                    level = {
                        "side": "BSL",
                        "price": price,
                        "pivot_time": df.index[pivot_position],
                        "swept_time": None,
                        "swept": False
                    }

                    liquidity_levels.append(level)
                    all_liquidity.append(level)

            if find_pivot(pivot_position, "low"):
                price = float(df["Low"].iloc[pivot_position])

                last_low = price
                last_low_broken = False

                if not any(
                    level["side"] == "SSL"
                    and level["price"] == price
                    for level in liquidity_levels
                ):
                    level = {
                        "side": "SSL",
                        "price": price,
                        "pivot_time": df.index[pivot_position],
                        "swept_time": None,
                        "swept": False
                    }

                    liquidity_levels.append(level)
                    all_liquidity.append(level)

        # 2. Detect liquidity sweeps.

        for level in liquidity_levels[:]:

            swept = (
                row["High"] > level["price"]
                if level["side"] == "BSL"
                else row["Low"] < level["price"]
            )

            if swept:
                level["swept"] = True
                level["swept_time"] = current_time

                df.iloc[
                    i,
                    df.columns.get_loc(
                        "BSL_Sweep" if level["side"] == "BSL"
                        else "SSL_Sweep"
                    )
                ] = True

                liquidity_levels.remove(level)

        if len(liquidity_levels) > cfg["max_liquidity_levels"]:
            liquidity_levels = liquidity_levels[
                -cfg["max_liquidity_levels"]:
            ]

        # 3. Detect confirmed BOS.

        previous_close = (
            df["Close"].iloc[i - 1] if i > 0 else None
        )

        bull_bos = (
            last_high is not None
            and not last_high_broken
            and previous_close is not None
            and previous_close <= last_high
            and row["Close"] > last_high
        )

        bear_bos = (
            last_low is not None
            and not last_low_broken
            and previous_close is not None
            and previous_close >= last_low
            and row["Close"] < last_low
        )

        if bull_bos:
            last_high_broken = True
            df.iloc[i, df.columns.get_loc("Bullish_BOS")] = True
            total_bullish_bos += 1

        if bear_bos:
            last_low_broken = True
            df.iloc[i, df.columns.get_loc("Bearish_BOS")] = True
            total_bearish_bos += 1

        # 4. Process a pending bullish FVG.

        if pending_bull is not None:

            age = i - pending_bull["bos_position"]

            if age > cfg["fvg_search"]:
                pending_bull = None

            elif i > pending_bull["bos_position"]:

                fvg = check_fvg(
                    i, 1,
                    pending_bull["ob_high"],
                    pending_bull["ob_low"]
                )

                if fvg:
                    create_setup(
                        1, pending_bull, fvg,
                        pending_bull["bos_time"]
                    )
                    df.iloc[
                        i, df.columns.get_loc("Bullish_Setup")
                    ] = True
                    pending_bull = None

        # 5. Process a pending bearish FVG.

        if pending_bear is not None:

            age = i - pending_bear["bos_position"]

            if age > cfg["fvg_search"]:
                pending_bear = None

            elif i > pending_bear["bos_position"]:

                fvg = check_fvg(
                    i, -1,
                    pending_bear["ob_high"],
                    pending_bear["ob_low"]
                )

                if fvg:
                    create_setup(
                        -1, pending_bear, fvg,
                        pending_bear["bos_time"]
                    )
                    df.iloc[
                        i, df.columns.get_loc("Bearish_Setup")
                    ] = True
                    pending_bear = None

        # 6. Find displacement and the opposing OB candle.

        for direction, bos in [(1, bull_bos), (-1, bear_bos)]:

            if not bos:
                continue

            displacement_position = None

            search_start = max(
                0, i - cfg["displacement_search"]
            )

            for k in range(i, search_start - 1, -1):

                candle = df.iloc[k]
                atr = candle["ATR"]

                if pd.isna(atr) or atr <= 0:
                    continue

                candle_range = candle["High"] - candle["Low"]

                if candle_range <= 0:
                    continue

                if direction == 1:
                    body = candle["Close"] - candle["Open"]
                    close_location = (
                        candle["Close"] - candle["Low"]
                    ) / candle_range
                else:
                    body = candle["Open"] - candle["Close"]
                    close_location = (
                        candle["High"] - candle["Close"]
                    ) / candle_range

                if (
                    body >= atr * cfg["min_displacement_atr"]
                    and close_location >= cfg["min_close_location"]
                ):
                    displacement_position = k
                    break

            if displacement_position is None:
                continue

            ob = None

            for j in range(
                displacement_position + 1,
                min(i + 1, displacement_position
                    + cfg["ob_lookback"] + 1)
            ):

                candle = df.iloc[j]
                atr = candle["ATR"]

                if pd.isna(atr) or atr <= 0:
                    continue

                candle_range = candle["High"] - candle["Low"]

                if direction == 1:
                    valid = (
                        candle["Close"] < candle["Open"]
                        and candle_range >= atr * cfg["min_ob_atr"]
                    )
                else:
                    valid = (
                        candle["Close"] > candle["Open"]
                        and candle_range >= atr * cfg["min_ob_atr"]
                    )

                if valid:
                    ob = {
                        "position": j,
                        "ob_time": df.index[j],
                        "ob_high": float(candle["High"]),
                        "ob_low": float(candle["Low"]),
                        "bos_position": i,
                        "bos_time": current_time
                    }
                    break

            if ob is None:
                continue

            # 7. Require an overlapping qualified FVG.

            fvg = find_setup(i, direction, ob)

            if fvg:
                create_setup(direction, ob, fvg, current_time)

                column = (
                    "Bullish_Setup" if direction == 1
                    else "Bearish_Setup"
                )

                df.iloc[i, df.columns.get_loc(column)] = True

            elif direction == 1:
                pending_bull = ob

            else:
                pending_bear = ob

        # 8. Invalidate paired zones on confirmed candle close.

        for zone in active_zones[:]:

            if zone["direction"] == "BULLISH":
                invalid = (
                    row["Close"] < zone["ob_low"]
                    or row["Close"] < zone["fvg_low"]
                )
            else:
                invalid = (
                    row["Close"] > zone["ob_high"]
                    or row["Close"] > zone["fvg_high"]
                )

            if invalid:
                zone["status"] = "INVALIDATED"
                active_zones.remove(zone)
                invalidated_count += 1

    # 9. Keep historical setup count within configured limit.

    setups = setups[-cfg["max_historical_setups"]:]

    setup_table = pd.DataFrame(setups)
    liquidity_table = pd.DataFrame(all_liquidity)

    statistics = {
        "total_qualified_setups": len(setups),
        "active_setups": sum(
            1 for s in setups if s["status"] == "ACTIVE"
        ),
        "invalidated_setups": invalidated_count,
        "bullish_bos": total_bullish_bos,
        "bearish_bos": total_bearish_bos,
        "active_bsl_levels": sum(
            1 for x in liquidity_levels if x["side"] == "BSL"
        ),
        "active_ssl_levels": sum(
            1 for x in liquidity_levels if x["side"] == "SSL"
        )
    }

    return {
        "candles": df,
        "setups": setup_table,
        "liquidity": liquidity_table,
        "statistics": statistics
    }

