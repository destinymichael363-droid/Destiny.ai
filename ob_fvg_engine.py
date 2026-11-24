from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class PairedZone:
    direction: Literal[1, -1]  # 1 = Bullish, -1 = Bearish
    ob_high: float
    ob_low: float
    fvg_high: float
    fvg_low: float
    ob_bar: int
    fvg_bar: int
    created_bar: int
    trend_alignment: Literal[
        "TREND ALIGNED",
        "COUNTERTREND",
        "NEUTRAL TREND",
    ]


@dataclass
class LiquidityLevel:
    price: float
    direction: Literal[1, -1]  # 1 = BSL, -1 = SSL
    pivot_bar: int

    # New state information.
    swept: bool = False
    swept_bar: int | None = None


def _rma(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)

    return series.ewm(
        alpha=1.0 / length,
        min_periods=length,
        adjust=False,
    ).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)

    return series.rolling(
        window=length,
        min_periods=length,
    ).mean()


def _atr(
    df: pd.DataFrame,
    length: int,
) -> pd.Series:
    high = pd.to_numeric(
        df["High"],
        errors="coerce",
    )

    low = pd.to_numeric(
        df["Low"],
        errors="coerce",
    )

    close = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return _rma(tr, length)


def _pivots_confirmed(
    series: pd.Series,
    left_len: int,
    right_len: int,
) -> pd.Series:
    """
    Confirmed pivot detector.

    A pivot at p is only reported at:

        p + right_len

    Therefore the pivot cannot appear before enough candles have
    closed to confirm it.

    This keeps the engine causal for backtesting/live execution.
    """

    vals = series.to_numpy(dtype=float)

    n = len(vals)

    out = np.full(
        n,
        np.nan,
        dtype=float,
    )

    if (
        left_len < 1
        or right_len < 0
        or n == 0
    ):
        return pd.Series(
            out,
            index=series.index,
        )

    for p in range(
        left_len,
        n - right_len,
    ):
        current = vals[p]

        if np.isnan(current):
            continue

        left_window = vals[
            p - left_len:p
        ]

        right_window = vals[
            p + 1:p + right_len + 1
        ]

        # High pivot.
        if (
            np.all(current > left_window)
            and np.all(current >= right_window)
        ):
            out[p + right_len] = current

    return pd.Series(
        out,
        index=series.index,
    )


def _pivots_confirmed_low(
    series: pd.Series,
    left_len: int,
    right_len: int,
) -> pd.Series:
    """
    Confirmed low pivot detector.

    Kept separate from the high detector so the comparison logic
    remains explicit and easy to audit.
    """

    vals = series.to_numpy(dtype=float)

    n = len(vals)

    out = np.full(
        n,
        np.nan,
        dtype=float,
    )

    if (
        left_len < 1
        or right_len < 0
        or n == 0
    ):
        return pd.Series(
            out,
            index=series.index,
        )

    for p in range(
        left_len,
        n - right_len,
    ):
        current = vals[p]

        if np.isnan(current):
            continue

        left_window = vals[
            p - left_len:p
        ]

        right_window = vals[
            p + 1:p + right_len + 1
        ]

        # Low pivot.
        if (
            np.all(current < left_window)
            and np.all(current <= right_window)
        ):
            out[p + right_len] = current

    return pd.Series(
        out,
        index=series.index,
    )


def _intervals_overlap(
    low_a: float,
    high_a: float,
    low_b: float,
    high_b: float,
) -> bool:
    """
    True when two price intervals genuinely overlap.

    Touching exactly at one boundary is not treated as overlap.
    """

    return max(
        low_a,
        low_b,
    ) < min(
        high_a,
        high_b,
    )


def _safe_float(
    value: float,
) -> float:
    try:
        value = float(value)
    except Exception:
        return np.nan

    return value if np.isfinite(value) else np.nan


def run_ob_fvg_engine(
    df: pd.DataFrame,
    timeframe_minutes: int | None = None,
    pivot_length: int = 5,
    max_liquidity_levels: int = 30,
    ob_lookback: int = 20,
    min_ob_atr: float = 0.35,
    displacement_search: int = 8,
    min_displacement_atr: float = 0.80,
    min_close_location: float = 0.65,
    fvg_search_bars: int = 24,
    atr_length: int = 14,
    min_fvg_atr: float = 0.20,
    use_volume_filter: bool = True,
    volume_length: int = 20,
    min_fvg_volume_ratio: float = 1.5,
    max_historical_setups: int = 10,
    base_dominant_length: int = 24,
    hourly_dominant_length: int = 24,
    volatility_lookback: int = 100,
    min_tick: float = 0.0,
) -> dict:
    """
    DESTINY AI OB + FVG structural engine.

    Preserves the original architecture:

        confirmed pivots
             ↓
        liquidity levels
             ↓
        dominant trend
             ↓
        liquidity sweep
             ↓
        BOS / CHoCH
             ↓
        displacement
             ↓
        order block
             ↓
        fair value gap
             ↓
        paired OB + FVG zone

    Important:
    This engine works only with information available at the current
    completed candle. It does not intentionally use future candles
    to create a signal.

    Returned structure remains compatible with the previous engine:

        {
            "active_zones": [...],
            "active_liquidity": [...],
            "signals_df": ...
        }
    """

    # ============================================================
    # 1. VALIDATION
    # ============================================================

    required = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
        or not all(
            column in df.columns
            for column in required
        )
    ):
        return {
            "active_zones": [],
            "active_liquidity": [],
            "signals_df": pd.DataFrame(),
        }

    data = df.copy()

    if "Volume" not in data.columns:
        data["Volume"] = 0.0

    # Numeric normalization.
    for column in required + ["Volume"]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=required
    )

    if data.empty:
        return {
            "active_zones": [],
            "active_liquidity": [],
            "signals_df": pd.DataFrame(),
        }

    data = data.reset_index(drop=False)

    # ============================================================
    # 2. PARAMETERS
    # ============================================================

    pivot_length = max(
        int(pivot_length),
        1,
    )

    max_liquidity_levels = max(
        int(max_liquidity_levels),
        1,
    )

    ob_lookback = max(
        int(ob_lookback),
        1,
    )

    displacement_search = max(
        int(displacement_search),
        1,
    )

    fvg_search_bars = max(
        int(fvg_search_bars),
        1,
    )

    atr_length = max(
        int(atr_length),
        1,
    )

    volume_length = max(
        int(volume_length),
        1,
    )

    max_historical_setups = max(
        int(max_historical_setups),
        1,
    )

    base_dominant_length = max(
        int(base_dominant_length),
        12,
    )

    hourly_dominant_length = max(
        int(hourly_dominant_length),
        12,
    )

    volatility_lookback = max(
        int(volatility_lookback),
        2,
    )

    min_displacement_atr = max(
        float(min_displacement_atr),
        0.0,
    )

    min_ob_atr = max(
        float(min_ob_atr),
        0.0,
    )

    min_fvg_atr = max(
        float(min_fvg_atr),
        0.0,
    )

    min_close_location = min(
        max(
            float(min_close_location),
            0.0,
        ),
        1.0,
    )

    min_fvg_volume_ratio = max(
        float(min_fvg_volume_ratio),
        0.0,
    )

    min_tick = max(
        float(min_tick),
        0.0,
    )

    # ============================================================
    # 3. ARRAYS
    # ============================================================

    opens = data["Open"].to_numpy(
        dtype=float
    )

    highs = data["High"].to_numpy(
        dtype=float
    )

    lows = data["Low"].to_numpy(
        dtype=float
    )

    closes = data["Close"].to_numpy(
        dtype=float
    )

    vols = data["Volume"].to_numpy(
        dtype=float
    )

    n = len(data)

    # ============================================================
    # 4. ATR / VOLUME
    # ============================================================

    atr_series = _atr(
        data,
        atr_length,
    )

    avg_vol_series = _sma(
        data["Volume"],
        volume_length,
    )

    atr_vals = atr_series.to_numpy(
        dtype=float
    )

    avg_vol_vals = avg_vol_series.to_numpy(
        dtype=float
    )

    # ============================================================
    # 5. CONFIRMED SWING PIVOTS
    # ============================================================

    swing_high_piv = (
        _pivots_confirmed(
            data["High"],
            pivot_length,
            pivot_length,
        )
        .to_numpy(dtype=float)
    )

    swing_low_piv = (
        _pivots_confirmed_low(
            data["Low"],
            pivot_length,
            pivot_length,
        )
        .to_numpy(dtype=float)
    )

    # ============================================================
    # 6. DYNAMIC DOMINANT STRUCTURE
    # ============================================================

    trend_atr_series = _atr(
        data,
        14,
    )

    avg_trend_atr_series = _sma(
        trend_atr_series,
        volatility_lookback,
    )

    trend_atr_vals = (
        trend_atr_series.to_numpy(
            dtype=float
        )
    )

    avg_trend_atr_vals = (
        avg_trend_atr_series.to_numpy(
            dtype=float
        )
    )

    dyn_high_piv = np.full(
        n,
        np.nan,
        dtype=float,
    )

    dyn_low_piv = np.full(
        n,
        np.nan,
        dtype=float,
    )

    dyn_lengths = np.full(
        n,
        base_dominant_length,
        dtype=int,
    )

    for t in range(n):
        # --------------------------------------------------------
        # Hourly dominant structure remains fixed.
        # --------------------------------------------------------
        if timeframe_minutes == 60:
            dom_len = int(
                hourly_dominant_length
            )

        else:
            t_atr = trend_atr_vals[t]
            avg_t_atr = avg_trend_atr_vals[t]

            if (
                np.isfinite(t_atr)
                and np.isfinite(avg_t_atr)
                and avg_t_atr > 0
            ):
                ratio = (
                    t_atr / avg_t_atr
                )
            else:
                ratio = 1.0

            ratio = max(
                0.75,
                min(
                    1.50,
                    ratio,
                ),
            )

            dom_len = int(
                round(
                    base_dominant_length
                    * ratio
                )
            )

            dom_len = max(
                12,
                min(
                    40,
                    dom_len,
                ),
            )

        dyn_lengths[t] = dom_len

        # --------------------------------------------------------
        # The pivot is evaluated only after dom_len right candles
        # have closed.
        # --------------------------------------------------------
        p = t - dom_len

        if p < dom_len:
            continue

        # Dominant high.
        c_h = highs[p]

        if np.isfinite(c_h):
            left_highs = highs[
                p - dom_len:p
            ]

            right_highs = highs[
                p + 1:t + 1
            ]

            if (
                np.all(c_h > left_highs)
                and np.all(c_h >= right_highs)
            ):
                dyn_high_piv[t] = c_h

        # Dominant low.
        c_l = lows[p]

        if np.isfinite(c_l):
            left_lows = lows[
                p - dom_len:p
            ]

            right_lows = lows[
                p + 1:t + 1
            ]

            if (
                np.all(c_l < left_lows)
                and np.all(c_l <= right_lows)
            ):
                dyn_low_piv[t] = c_l

    # ============================================================
    # 7. STRUCTURAL STATE
    # ============================================================

    last_swing_high = np.nan
    last_swing_low = np.nan

    last_swing_high_bar: int | None = None
    last_swing_low_bar: int | None = None

    last_swing_high_broken = False
    last_swing_low_broken = False

    # Dominant trend pivots.
    prev_trend_high = np.nan
    latest_trend_high = np.nan

    prev_trend_low = np.nan
    latest_trend_low = np.nan

    prev_trend_high_bar: int | None = None
    latest_trend_high_bar: int | None = None

    prev_trend_low_bar: int | None = None
    latest_trend_low_bar: int | None = None

    trendline_broken = False
    trendline_direction = 0

    # This is the structural direction established by BOS/CHoCH.
    #
    #  1 = bullish
    # -1 = bearish
    #  0 = unknown
    structure_bias = 0

    # ============================================================
    # 8. LIQUIDITY / ZONES
    # ============================================================

    liq_levels: list[
        LiquidityLevel
    ] = []

    zones: list[
        PairedZone
    ] = []

    # ============================================================
    # 9. RECORDS
    # ============================================================

    records: list[dict] = []

    # ============================================================
    # 10. MAIN CAUSAL BAR LOOP
    # ============================================================

    for i in range(n):

        o = opens[i]
        h = highs[i]
        l = lows[i]
        c = closes[i]

        prev_c = (
            closes[i - 1]
            if i > 0
            else np.nan
        )

        # --------------------------------------------------------
        # A. CONFIRMED SWING HIGH
        # --------------------------------------------------------

        if np.isfinite(
            swing_high_piv[i]
        ):
            last_swing_high = float(
                swing_high_piv[i]
            )

            last_swing_high_bar = (
                i - pivot_length
            )

            last_swing_high_broken = False

            duplicate = any(
                lv.direction == 1
                and (
                    abs(
                        lv.price
                        - last_swing_high
                    )
                    <= min_tick
                )
                for lv in liq_levels
            )

            if not duplicate:
                liq_levels.append(
                    LiquidityLevel(
                        price=last_swing_high,
                        direction=1,
                        pivot_bar=(
                            last_swing_high_bar
                        ),
                    )
                )

        # --------------------------------------------------------
        # B. CONFIRMED SWING LOW
        # --------------------------------------------------------

        if np.isfinite(
            swing_low_piv[i]
        ):
            last_swing_low = float(
                swing_low_piv[i]
            )

            last_swing_low_bar = (
                i - pivot_length
            )

            last_swing_low_broken = False

            duplicate = any(
                lv.direction == -1
                and (
                    abs(
                        lv.price
                        - last_swing_low
                    )
                    <= min_tick
                )
                for lv in liq_levels
            )

            if not duplicate:
                liq_levels.append(
                    LiquidityLevel(
                        price=last_swing_low,
                        direction=-1,
                        pivot_bar=(
                            last_swing_low_bar
                        ),
                    )
                )

        # --------------------------------------------------------
        # Keep only the configured number of liquidity levels.
        #
        # Newest levels are retained.
        # --------------------------------------------------------

        if (
            len(liq_levels)
            > max_liquidity_levels
        ):
            liq_levels = liq_levels[
                -max_liquidity_levels:
            ]

        # --------------------------------------------------------
        # C. DOMINANT TREND PIVOTS
        # --------------------------------------------------------

        dom_len = int(
            dyn_lengths[i]
        )

        if np.isfinite(
            dyn_high_piv[i]
        ):
            prev_trend_high = (
                latest_trend_high
            )

            prev_trend_high_bar = (
                latest_trend_high_bar
            )

            latest_trend_high = float(
                dyn_high_piv[i]
            )

            latest_trend_high_bar = (
                i - dom_len
            )

        if np.isfinite(
            dyn_low_piv[i]
        ):
            prev_trend_low = (
                latest_trend_low
            )

            prev_trend_low_bar = (
                latest_trend_low_bar
            )

            latest_trend_low = float(
                dyn_low_piv[i]
            )

            latest_trend_low_bar = (
                i - dom_len
            )

        # --------------------------------------------------------
        # D. DOMINANT STRUCTURE
        # --------------------------------------------------------

        has_all_four = not any(
            np.isnan(value)
            for value in (
                prev_trend_high,
                latest_trend_high,
                prev_trend_low,
                latest_trend_low,
            )
        )

        bull_struct = (
            has_all_four
            and latest_trend_high
            > prev_trend_high
            and latest_trend_low
            > prev_trend_low
        )

        bear_struct = (
            has_all_four
            and latest_trend_high
            < prev_trend_high
            and latest_trend_low
            < prev_trend_low
        )

        if bull_struct and not bear_struct:
            raw_trend_dir = 1

        elif bear_struct and not bull_struct:
            raw_trend_dir = -1

        else:
            raw_trend_dir = 0

        # --------------------------------------------------------
        # E. TRENDLINE STATE
        # --------------------------------------------------------

        new_bull_tl = (
            np.isfinite(
                dyn_low_piv[i]
            )
            and np.isfinite(
                prev_trend_low
            )
            and prev_trend_low_bar
            is not None
            and latest_trend_low
            > prev_trend_low
        )

        new_bear_tl = (
            np.isfinite(
                dyn_high_piv[i]
            )
            and np.isfinite(
                prev_trend_high
            )
            and prev_trend_high_bar
            is not None
            and latest_trend_high
            < prev_trend_high
        )

        est_bull = (
            new_bull_tl
            and not new_bear_tl
        )

        est_bear = (
            new_bear_tl
            and not new_bull_tl
        )

        if (
            new_bull_tl
            and new_bear_tl
        ):
            est_bull = (
                raw_trend_dir == 1
            )

            est_bear = (
                raw_trend_dir == -1
            )

        if est_bull:
            trendline_direction = 1
            trendline_broken = False

        elif est_bear:
            trendline_direction = -1
            trendline_broken = False

        # --------------------------------------------------------
        # F. TRENDLINE PROJECTION
        # --------------------------------------------------------

        proj_price = np.nan

        if (
            trendline_direction == 1
            and prev_trend_low_bar
            is not None
            and latest_trend_low_bar
            is not None
            and latest_trend_low_bar
            > prev_trend_low_bar
            and np.isfinite(
                prev_trend_low
            )
            and np.isfinite(
                latest_trend_low
            )
        ):
            slope = (
                latest_trend_low
                - prev_trend_low
            ) / float(
                latest_trend_low_bar
                - prev_trend_low_bar
            )

            proj_price = (
                latest_trend_low
                + slope
                * float(
                    i
                    - latest_trend_low_bar
                )
            )

        elif (
            trendline_direction == -1
            and prev_trend_high_bar
            is not None
            and latest_trend_high_bar
            is not None
            and latest_trend_high_bar
            > prev_trend_high_bar
            and np.isfinite(
                prev_trend_high
            )
            and np.isfinite(
                latest_trend_high
            )
        ):
            slope = (
                latest_trend_high
                - prev_trend_high
            ) / float(
                latest_trend_high_bar
                - prev_trend_high_bar
            )

            proj_price = (
                latest_trend_high
                + slope
                * float(
                    i
                    - latest_trend_high_bar
                )
            )

        # --------------------------------------------------------
        # G. TRENDLINE BREAK
        # --------------------------------------------------------

        trendline_break_event = False

        if (
            not trendline_broken
            and np.isfinite(proj_price)
        ):
            if (
                trendline_direction == 1
                and c < proj_price
            ):
                trendline_broken = True
                trendline_break_event = True

            elif (
                trendline_direction == -1
                and c > proj_price
            ):
                trendline_broken = True
                trendline_break_event = True

        trend_dir = (
            0
            if trendline_broken
            else raw_trend_dir
        )

        # --------------------------------------------------------
        # H. LIQUIDITY SWEEP
        # --------------------------------------------------------
        #
        # IMPORTANT CHANGE:
        #
        # A wick through BSL/SSL is now recorded as a sweep.
        # It is NOT automatically treated as a permanent structural
        # breakout.
        #
        # The level is marked swept and retained long enough for the
        # current candle to be classified correctly.
        # --------------------------------------------------------

        swept_count = 0
        swept_bsl_count = 0
        swept_ssl_count = 0

        swept_bsl_prices: list[
            float
        ] = []

        swept_ssl_prices: list[
            float
        ] = []

        for lv in liq_levels:

            if lv.swept:
                continue

            if (
                lv.direction == 1
                and h > lv.price
            ):
                lv.swept = True
                lv.swept_bar = i

                swept_count += 1
                swept_bsl_count += 1

                swept_bsl_prices.append(
                    float(lv.price)
                )

            elif (
                lv.direction == -1
                and l < lv.price
            ):
                lv.swept = True
                lv.swept_bar = i

                swept_count += 1
                swept_ssl_count += 1

                swept_ssl_prices.append(
                    float(lv.price)
                )

        # --------------------------------------------------------
        # I. BOS DETECTION
        # --------------------------------------------------------
        #
        # BOS requires a CLOSED candle to cross the confirmed swing
        # level.
        #
        # A wick alone is not BOS.
        # --------------------------------------------------------

        crossed_above = (
            i > 0
            and np.isfinite(
                last_swing_high
            )
            and np.isfinite(prev_c)
            and prev_c
            <= last_swing_high
            and c
            > last_swing_high
        )

        crossed_below = (
            i > 0
            and np.isfinite(
                last_swing_low
            )
            and np.isfinite(prev_c)
            and prev_c
            >= last_swing_low
            and c
            < last_swing_low
        )

        bull_bos = (
            not last_swing_high_broken
            and np.isfinite(
                last_swing_high
            )
            and crossed_above
        )

        bear_bos = (
            not last_swing_low_broken
            and np.isfinite(
                last_swing_low
            )
            and crossed_below
        )

        # --------------------------------------------------------
        # J. BOS / CHoCH CLASSIFICATION
        # --------------------------------------------------------

        bos_level = np.nan
        bos_bar = -1

        choch_bull = False
        choch_bear = False

        structure_event = ""

        if bull_bos:

            last_swing_high_broken = True

            bos_level = float(
                last_swing_high
            )

            bos_bar = (
                int(last_swing_high_bar)
                if last_swing_high_bar
                is not None
                else i
            )

            # If the previous structural state was bearish,
            # this bullish break is a CHoCH.
            if structure_bias == -1:
                choch_bull = True
                structure_event = "BULLISH CHOCH"

            else:
                structure_event = "BULLISH BOS"

            structure_bias = 1

        if bear_bos:

            last_swing_low_broken = True

            bos_level = float(
                last_swing_low
            )

            bos_bar = (
                int(last_swing_low_bar)
                if last_swing_low_bar
                is not None
                else i
            )

            if structure_bias == 1:
                choch_bear = True
                structure_event = "BEARISH CHOCH"

            else:
                structure_event = "BEARISH BOS"

            structure_bias = -1

        # --------------------------------------------------------
        # K. HELPER FOR TREND ALIGNMENT
        # --------------------------------------------------------

        def _align_label(
            direction: int,
        ) -> Literal[
            "TREND ALIGNED",
            "COUNTERTREND",
            "NEUTRAL TREND",
        ]:

            if trend_dir == direction:
                return "TREND ALIGNED"

            if trend_dir == -direction:
                return "COUNTERTREND"

            return "NEUTRAL TREND"

        # ========================================================
        # L. BULLISH DISPLACEMENT -> OB -> FVG
        # ========================================================

        new_bull_setup = False

        if bull_bos:

            disp_offset: int | None = None

            # The displacement is allowed to be the BOS candle or
            # one of the immediately preceding candles.
            #
            # This preserves your original structural concept:
            # displacement is what creates the actual break.
            #
            for k in range(
                0,
                min(
                    displacement_search,
                    i,
                )
                + 1,
            ):

                idx_k = i - k

                rng = (
                    highs[idx_k]
                    - lows[idx_k]
                )

                body = (
                    closes[idx_k]
                    - opens[idx_k]
                )

                a_k = atr_vals[idx_k]

                if (
                    not np.isfinite(a_k)
                    or a_k <= 0
                    or rng <= 0
                ):
                    continue

                close_location = (
                    closes[idx_k]
                    - lows[idx_k]
                ) / rng

                if (
                    body
                    >= a_k
                    * min_displacement_atr
                    and close_location
                    >= min_close_location
                ):
                    disp_offset = k
                    break

            # ----------------------------------------------------
            # Find the bearish candle before displacement.
            # ----------------------------------------------------

            ob_offset: int | None = None

            if disp_offset is not None:

                for j in range(
                    disp_offset + 1,
                    min(
                        ob_lookback,
                        i,
                    )
                    + 1,
                ):

                    idx_j = i - j

                    a_j = atr_vals[idx_j]

                    if (
                        not np.isfinite(a_j)
                        or a_j <= 0
                    ):
                        continue

                    candle_range = (
                        highs[idx_j]
                        - lows[idx_j]
                    )

                    if (
                        closes[idx_j]
                        < opens[idx_j]
                        and candle_range
                        >= a_j
                        * min_ob_atr
                    ):
                        ob_offset = j
                        break

            # ----------------------------------------------------
            # Find overlapping bullish FVG.
            # ----------------------------------------------------

            if ob_offset is not None:

                ob_idx = (
                    i - ob_offset
                )

                bull_ob_high = float(
                    highs[ob_idx]
                )

                bull_ob_low = float(
                    lows[ob_idx]
                )

                matched_fvg_offset: int | None = None

                matched_fvg_high = np.nan
                matched_fvg_low = np.nan

                for f_off in range(
                    0,
                    min(
                        fvg_search_bars,
                        i,
                    )
                    + 1,
                ):

                    idx_f = i - f_off

                    if idx_f - 2 < 0:
                        continue

                    # FVG must be associated with the displacement
                    # sequence rather than an earlier unrelated gap.
                    if (
                        f_off
                        >= ob_offset
                    ):
                        continue

                    vol_idx = idx_f - 1

                    if use_volume_filter:

                        volume_pass = (
                            vol_idx >= 0
                            and np.isfinite(
                                avg_vol_vals[
                                    vol_idx
                                ]
                            )
                            and vols[vol_idx]
                            >= avg_vol_vals[
                                vol_idx
                            ]
                            * min_fvg_volume_ratio
                        )

                    else:
                        volume_pass = True

                    # Bullish FVG:
                    #
                    # current low > high two candles earlier
                    gap_low = float(
                        highs[idx_f - 2]
                    )

                    gap_high = float(
                        lows[idx_f]
                    )

                    a_f = atr_vals[idx_f]

                    if (
                        not np.isfinite(a_f)
                        or a_f <= 0
                    ):
                        continue

                    gap_size = (
                        gap_high
                        - gap_low
                    )

                    if (
                        gap_high
                        > gap_low
                        and gap_size
                        >= a_f
                        * min_fvg_atr
                        and volume_pass
                        and _intervals_overlap(
                            bull_ob_low,
                            bull_ob_high,
                            gap_low,
                            gap_high,
                        )
                    ):
                        matched_fvg_offset = (
                            f_off
                        )

                        matched_fvg_high = (
                            gap_high
                        )

                        matched_fvg_low = (
                            gap_low
                        )

                        break

                if (
                    matched_fvg_offset
                    is not None
                ):

                    zones.append(
                        PairedZone(
                            direction=1,
                            ob_high=bull_ob_high,
                            ob_low=bull_ob_low,
                            fvg_high=float(
                                matched_fvg_high
                            ),
                            fvg_low=float(
                                matched_fvg_low
                            ),
                            ob_bar=ob_idx,
                            fvg_bar=(
                                i
                                - matched_fvg_offset
                            ),
                            created_bar=i,
                            trend_alignment=(
                                _align_label(1)
                            ),
                        )
                    )

                    new_bull_setup = True

        # ========================================================
        # M. BEARISH DISPLACEMENT -> OB -> FVG
        # ========================================================

        new_bear_setup = False

        if bear_bos:

            disp_offset: int | None = None

            for k in range(
                0,
                min(
                    displacement_search,
                    i,
                )
                + 1,
            ):

                idx_k = i - k

                rng = (
                    highs[idx_k]
                    - lows[idx_k]
                )

                body = (
                    opens[idx_k]
                    - closes[idx_k]
                )

                a_k = atr_vals[idx_k]

                if (
                    not np.isfinite(a_k)
                    or a_k <= 0
                    or rng <= 0
                ):
                    continue

                close_location = (
                    highs[idx_k]
                    - closes[idx_k]
                ) / rng

                if (
                    body
                    >= a_k
                    * min_displacement_atr
                    and close_location
                    >= min_close_location
                ):
                    disp_offset = k
                    break

            # ----------------------------------------------------
            # Find bullish candle before bearish displacement.
            # ----------------------------------------------------

            ob_offset: int | None = None

            if disp_offset is not None:

                for j in range(
                    disp_offset + 1,
                    min(
                        ob_lookback,
                        i,
                    )
                    + 1,
                ):

                    idx_j = i - j

                    a_j = atr_vals[idx_j]

                    if (
                        not np.isfinite(a_j)
                        or a_j <= 0
                    ):
                        continue

                    candle_range = (
                        highs[idx_j]
                        - lows[idx_j]
                    )

                    if (
                        closes[idx_j]
                        > opens[idx_j]
                        and candle_range
                        >= a_j
                        * min_ob_atr
                    ):
                        ob_offset = j
                        break

            # ----------------------------------------------------
            # Find overlapping bearish FVG.
            # ----------------------------------------------------

            if ob_offset is not None:

                ob_idx = (
                    i - ob_offset
                )

                bear_ob_high = float(
                    highs[ob_idx]
                )

                bear_ob_low = float(
                    lows[ob_idx]
                )

                matched_fvg_offset: int | None = None

                matched_fvg_high = np.nan
                matched_fvg_low = np.nan

                for f_off in range(
                    0,
                    min(
                        fvg_search_bars,
                        i,
                    )
                    + 1,
                ):

                    idx_f = i - f_off

                    if idx_f - 2 < 0:
                        continue

                    if (
                        f_off
                        >= ob_offset
                    ):
                        continue

                    vol_idx = idx_f - 1

                    if use_volume_filter:

                        volume_pass = (
                            vol_idx >= 0
                            and np.isfinite(
                                avg_vol_vals[
                                    vol_idx
                                ]
                            )
                            and vols[vol_idx]
                            >= avg_vol_vals[
                                vol_idx
                            ]
                            * min_fvg_volume_ratio
                        )

                    else:
                        volume_pass = True

                    # Bearish FVG:
                    #
                    # current high < low two candles earlier
                    gap_high = float(
                        lows[idx_f - 2]
                    )

                    gap_low = float(
                        highs[idx_f]
                    )

                    a_f = atr_vals[idx_f]

                    if (
                        not np.isfinite(a_f)
                        or a_f <= 0
                    ):
                        continue

                    gap_size = (
                        gap_high
                        - gap_low
                    )

                    if (
                        gap_high
                        > gap_low
                        and gap_size
                        >= a_f
                        * min_fvg_atr
                        and volume_pass
                        and _intervals_overlap(
                            bear_ob_low,
                            bear_ob_high,
                            gap_low,
                            gap_high,
                        )
                    ):
                        matched_fvg_offset = (
                            f_off
                        )

                        matched_fvg_high = (
                            gap_high
                        )

                        matched_fvg_low = (
                            gap_low
                        )

                        break

                if (
                    matched_fvg_offset
                    is not None
                ):

                    zones.append(
                        PairedZone(
                            direction=-1,
                            ob_high=bear_ob_high,
                            ob_low=bear_ob_low,
                            fvg_high=float(
                                matched_fvg_high
                            ),
                            fvg_low=float(
                                matched_fvg_low
                            ),
                            ob_bar=ob_idx,
                            fvg_bar=(
                                i
                                - matched_fvg_offset
                            ),
                            created_bar=i,
                            trend_alignment=(
                                _align_label(-1)
                            ),
                        )
                    )

                    new_bear_setup = True

        # ========================================================
        # N. CURRENT FVG DETECTION
        # ========================================================

        vol_prev_ok = True

        if use_volume_filter:

            vol_prev_ok = (
                i - 1 >= 0
                and np.isfinite(
                    avg_vol_vals[i - 1]
                )
                and vols[i - 1]
                >= avg_vol_vals[i - 1]
                * min_fvg_volume_ratio
            )

        curr_bull_fvg = False
        curr_bear_fvg = False

        if (
            i - 2 >= 0
            and np.isfinite(
                atr_vals[i]
            )
            and atr_vals[i] > 0
            and vol_prev_ok
        ):

            bullish_gap = (
                l
                - highs[i - 2]
            )

            bearish_gap = (
                lows[i - 2]
                - h
            )

            curr_bull_fvg = (
                bullish_gap > 0
                and bullish_gap
                >= atr_vals[i]
                * min_fvg_atr
            )

            curr_bear_fvg = (
                bearish_gap > 0
                and bearish_gap
                >= atr_vals[i]
                * min_fvg_atr
            )

        # ========================================================
        # O. ZONE INVALIDATION
        # ========================================================
        #
        # Close-based invalidation:
        #
        # Bullish zone:
        #     close below OB/FVG
        #
        # Bearish zone:
        #     close above OB/FVG
        #
        # A wick alone does not automatically destroy the zone.
        # This is intentional and consistent with the S/R engine.
        # ========================================================

        invalidated_count = 0

        for idx in range(
            len(zones) - 1,
            -1,
            -1,
        ):

            z = zones[idx]

            if z.direction == 1:

                invalid = (
                    c < z.ob_low
                    or c < z.fvg_low
                )

            else:

                invalid = (
                    c > z.ob_high
                    or c > z.fvg_high
                )

            if invalid:
                zones.pop(idx)
                invalidated_count += 1

        # ========================================================
        # P. HISTORICAL ZONE LIMIT
        # ========================================================

        while (
            len(zones)
            > max_historical_setups
        ):
            zones.pop(0)

        # ========================================================
        # Q. LIQUIDITY CLEANUP
        # ========================================================
        #
        # Swept levels are retained for the current structural
        # interpretation, then removed on the following cycle.
        #
        # This allows the signal record to know:
        #
        #     "liquidity was swept"
        #
        # without permanently treating the wick as a breakout.
        # ========================================================

        liq_levels = [
            lv
            for lv in liq_levels
            if not (
                lv.swept
                and lv.swept_bar is not None
                and lv.swept_bar < i
            )
        ]

        # ========================================================
        # R. SIGNAL RECORD
        # ========================================================

        records.append(
            {
                "timestamp": data.index[i],

                # Existing fields.
                "trend_direction": trend_dir,

                "bull_bos": bull_bos,
                "bear_bos": bear_bos,

                "bos_level": bos_level,
                "bos_bar": bos_bar,

                "new_bull_setup": (
                    new_bull_setup
                ),

                "new_bear_setup": (
                    new_bear_setup
                ),

                "swept_liquidity_count": (
                    swept_count
                ),

                "invalidated_zones_count": (
                    invalidated_count
                ),

                "active_zones_count": (
                    len(zones)
                ),

                "active_liquidity_count": (
                    len(liq_levels)
                ),

                # New structural fields.
                "choch_bull": (
                    choch_bull
                ),

                "choch_bear": (
                    choch_bear
                ),

                "bull_choch": (
                    choch_bull
                ),

                "bear_choch": (
                    choch_bear
                ),

                "structure_bias": (
                    structure_bias
                ),

                "structure_event": (
                    structure_event
                ),

                "trendline_break": (
                    trendline_break_event
                ),

                "swept_bsl_count": (
                    swept_bsl_count
                ),

                "swept_ssl_count": (
                    swept_ssl_count
                ),

                "sweep_bsl": (
                    swept_bsl_count > 0
                ),

                "sweep_ssl": (
                    swept_ssl_count > 0
                ),

                "swept_bsl_prices": (
                    tuple(swept_bsl_prices)
                ),

                "swept_ssl_prices": (
                    tuple(swept_ssl_prices)
                ),

                "displacement_bull": (
                    new_bull_setup
                ),

                "displacement_bear": (
                    new_bear_setup
                ),
            }
        )

    # ============================================================
    # 11. BUILD SIGNAL DATAFRAME
    # ============================================================

    signals_df = pd.DataFrame(
        records
    )

    if not signals_df.empty:
        signals_df = (
            signals_df
            .set_index("timestamp")
        )

    # ============================================================
    # 12. FINAL OUTPUT
    # ============================================================

    return {
        "active_zones": zones,
        "active_liquidity": liq_levels,
        "signals_df": signals_df,
    }
