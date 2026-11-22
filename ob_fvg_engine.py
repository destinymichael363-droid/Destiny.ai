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
    trend_alignment: Literal["TREND ALIGNED", "COUNTERTREND", "NEUTRAL TREND"]


@dataclass
class LiquidityLevel:
    price: float
    direction: Literal[1, -1]  # 1 = BSL, -1 = SSL
    pivot_bar: int


def _rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()


def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length, min_periods=length).mean()


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
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


def _pivots_confirmed(series: pd.Series, left_len: int, right_len: int) -> pd.Series:
    """
    Returns confirmed pivot values placed at index `p + right_len` (zero lookahead).
    """
    vals = series.to_numpy(dtype=float)
    n = len(vals)
    out = np.full(n, np.nan, dtype=float)
    if left_len < 1 or right_len < 0 or n == 0:
        return pd.Series(out, index=series.index)

    for p in range(left_len, n - right_len):
        c = vals[p]
        if np.isnan(c):
            continue
        left_w = vals[p - left_len : p]
        right_w = vals[p + 1 : p + right_len + 1]
        if np.all(c > left_w) and np.all(c >= right_w):
            out[p + right_len] = c
    return pd.Series(out, index=series.index)


def _intervals_overlap(low_a: float, high_a: float, low_b: float, high_b: float) -> bool:
    return max(low_a, low_b) < min(high_a, high_b)


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
    Translates 'Destiny OB + FVG Trend Edition' bar-by-bar without lookahead.
    Expects DataFrame with Open, High, Low, Close (and Volume if use_volume_filter=True).
    """
    required = ["Open", "High", "Low", "Close"]
    if df is None or df.empty or not all(col in df.columns for col in required):
        return {"active_zones": [], "active_liquidity": [], "signals_df": pd.DataFrame()}

    data = df.copy()
    if "Volume" not in data.columns:
        data["Volume"] = 0.0

    opens = data["Open"].to_numpy(dtype=float)
    highs = data["High"].to_numpy(dtype=float)
    lows = data["Low"].to_numpy(dtype=float)
    closes = data["Close"].to_numpy(dtype=float)
    vols = data["Volume"].to_numpy(dtype=float)

    atr_series = _atr(data, atr_length)
    avg_vol_series = _sma(data["Volume"].astype(float), volume_length)
    atr_vals = atr_series.to_numpy(dtype=float)
    avg_vol_vals = avg_vol_series.to_numpy(dtype=float)

    swing_high_piv = _pivots_confirmed(data["High"], pivot_length, pivot_length).to_numpy(dtype=float)
    swing_low_piv = _pivots_confirmed(data["Low"], pivot_length, pivot_length).to_numpy(dtype=float)

    # Precompute dynamic dominant trend pivots per bar (matches variable dominantSwingLength in Pine)
    trend_atr_series = _atr(data, 14)
    avg_trend_atr_series = _sma(trend_atr_series, volatility_lookback)
    trend_atr_vals = trend_atr_series.to_numpy(dtype=float)
    avg_trend_atr_vals = avg_trend_atr_series.to_numpy(dtype=float)

    n = len(data)
    dyn_high_piv = np.full(n, np.nan, dtype=float)
    dyn_low_piv = np.full(n, np.nan, dtype=float)
    dyn_lengths = np.full(n, base_dominant_length, dtype=int)

    for t in range(n):
        if timeframe_minutes == 60:
            dom_len = int(hourly_dominant_length)
        else:
            t_atr = trend_atr_vals[t]
            avg_t_atr = avg_trend_atr_vals[t]
            ratio = (t_atr / avg_t_atr) if (not np.isnan(t_atr) and not np.isnan(avg_t_atr) and avg_t_atr > 0) else 1.0
            clamped_ratio = max(0.75, min(1.50, ratio))
            dom_len = int(max(12, min(40, round(base_dominant_length * clamped_ratio))))
        dyn_lengths[t] = dom_len

        p = t - dom_len
        if p >= dom_len:
            c_h = highs[p]
            if not np.isnan(c_h):
                if np.all(c_h > highs[p - dom_len : p]) and np.all(c_h >= highs[p + 1 : t + 1]):
                    dyn_high_piv[t] = c_h

            c_l = lows[p]
            if not np.isnan(c_l):
                if np.all(c_l < lows[p - dom_len : p]) and np.all(c_l <= lows[p + 1 : t + 1]):
                    dyn_low_piv[t] = c_l

    # State variables
    last_swing_high = np.nan
    last_swing_low = np.nan
    last_swing_high_bar: int | None = None
    last_swing_low_bar: int | None = None
    last_swing_high_broken = False
    last_swing_low_broken = False

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

    liq_levels: list[LiquidityLevel] = []
    zones: list[PairedZone] = []

    bull_pending = False
    bull_cand_high = np.nan
    bull_cand_low = np.nan
    bull_cand_bar: int | None = None
    bull_pending_since: int | None = None

    bear_pending = False
    bear_cand_high = np.nan
    bear_cand_low = np.nan
    bear_cand_bar: int | None = None
    bear_pending_since: int | None = None

    records: list[dict] = []

    for i in range(n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        prev_c = closes[i - 1] if i > 0 else np.nan

        # 4. Swing pivots & BSL/SSL
        if not np.isnan(swing_high_piv[i]):
            last_swing_high = float(swing_high_piv[i])
            last_swing_high_bar = i - pivot_length
            last_swing_high_broken = False

            dup = any(
                lv.direction == 1 and abs(lv.price - last_swing_high) <= min_tick
                for lv in liq_levels
            )
            if not dup:
                liq_levels.append(LiquidityLevel(price=last_swing_high, direction=1, pivot_bar=last_swing_high_bar))
                if len(liq_levels) > max_liquidity_levels:
                    liq_levels.pop(0)

        if not np.isnan(swing_low_piv[i]):
            last_swing_low = float(swing_low_piv[i])
            last_swing_low_bar = i - pivot_length
            last_swing_low_broken = False

            dup = any(
                lv.direction == -1 and abs(lv.price - last_swing_low) <= min_tick
                for lv in liq_levels
            )
            if not dup:
                liq_levels.append(LiquidityLevel(price=last_swing_low, direction=-1, pivot_bar=last_swing_low_bar))
                if len(liq_levels) > max_liquidity_levels:
                    liq_levels.pop(0)

        # 4B. Dominant Trend Engine
        dom_len = int(dyn_lengths[i])
        if not np.isnan(dyn_high_piv[i]):
            prev_trend_high = latest_trend_high
            prev_trend_high_bar = latest_trend_high_bar
            latest_trend_high = float(dyn_high_piv[i])
            latest_trend_high_bar = i - dom_len

        if not np.isnan(dyn_low_piv[i]):
            prev_trend_low = latest_trend_low
            prev_trend_low_bar = latest_trend_low_bar
            latest_trend_low = float(dyn_low_piv[i])
            latest_trend_low_bar = i - dom_len

        has_all_four = not (
            np.isnan(prev_trend_high)
            or np.isnan(latest_trend_high)
            or np.isnan(prev_trend_low)
            or np.isnan(latest_trend_low)
        )
        bull_struct = has_all_four and (latest_trend_high > prev_trend_high) and (latest_trend_low > prev_trend_low)
        bear_struct = has_all_four and (latest_trend_high < prev_trend_high) and (latest_trend_low < prev_trend_low)

        raw_trend_dir = 1 if (bull_struct and not bear_struct) else (-1 if (bear_struct and not bull_struct) else 0)

        new_bull_tl = (not np.isnan(dyn_low_piv[i])) and (not np.isnan(prev_trend_low)) and (prev_trend_low_bar is not None) and (latest_trend_low > prev_trend_low)
        new_bear_tl = (not np.isnan(dyn_high_piv[i])) and (not np.isnan(prev_trend_high)) and (prev_trend_high_bar is not None) and (latest_trend_high < prev_trend_high)

        est_bull = new_bull_tl and not new_bear_tl
        est_bear = new_bear_tl and not new_bull_tl
        if new_bull_tl and new_bear_tl:
            est_bull = raw_trend_dir == 1
            est_bear = raw_trend_dir == -1

        if est_bull:
            trendline_direction = 1
            trendline_broken = False
        elif est_bear:
            trendline_direction = -1
            trendline_broken = False

        proj_price = np.nan
        if (
            trendline_direction == 1
            and prev_trend_low_bar is not None
            and latest_trend_low_bar is not None
            and latest_trend_low_bar > prev_trend_low_bar
            and not np.isnan(prev_trend_low)
            and not np.isnan(latest_trend_low)
        ):
            slope = (latest_trend_low - prev_trend_low) / float(latest_trend_low_bar - prev_trend_low_bar)
            proj_price = latest_trend_low + slope * float(i - latest_trend_low_bar)
        elif (
            trendline_direction == -1
            and prev_trend_high_bar is not None
            and latest_trend_high_bar is not None
            and latest_trend_high_bar > prev_trend_high_bar
            and not np.isnan(prev_trend_high)
            and not np.isnan(latest_trend_high)
        ):
            slope = (latest_trend_high - prev_trend_high) / float(latest_trend_high_bar - prev_trend_high_bar)
            proj_price = latest_trend_high + slope * float(i - latest_trend_high_bar)

        trendline_break_event = False
        if not trendline_broken and not np.isnan(proj_price):
            if (trendline_direction == 1 and c < proj_price) or (trendline_direction == -1 and c > proj_price):
                trendline_broken = True
                trendline_break_event = True

        trend_dir = 0 if trendline_broken else raw_trend_dir

        # 5. Liquidity sweep detection
        swept_count = 0
        for idx in range(len(liq_levels) - 1, -1, -1):
            lv = liq_levels[idx]
            swept = (h > lv.price) if lv.direction == 1 else (l < lv.price)
            if swept:
                liq_levels.pop(idx)
                swept_count += 1

        # 6. BOS detection (crossover / crossunder of close)
        crossed_above = (
            i > 0
            and not np.isnan(last_swing_high)
            and not np.isnan(prev_c)
            and (prev_c <= last_swing_high)
            and (c > last_swing_high)
        )
        crossed_below = (
            i > 0
            and not np.isnan(last_swing_low)
            and not np.isnan(prev_c)
            and (prev_c >= last_swing_low)
            and (c < last_swing_low)
        )

        bull_bos = (not last_swing_high_broken) and (not np.isnan(last_swing_high)) and crossed_above
        bear_bos = (not last_swing_low_broken) and (not np.isnan(last_swing_low)) and crossed_below

        if bull_bos:
            last_swing_high_broken = True
        if bear_bos:
            last_swing_low_broken = True

        new_bull_setup = False
        new_bear_setup = False

        def _align_label(dir_val: int) -> Literal["TREND ALIGNED", "COUNTERTREND", "NEUTRAL TREND"]:
            if trend_dir == dir_val:
                return "TREND ALIGNED"
            if trend_dir == -dir_val:
                return "COUNTERTREND"
            return "NEUTRAL TREND"

        # 10. Bullish BOS -> Displacement -> OB -> Overlapping FVG
        if bull_bos:
            bull_pending = False
            disp_offset: int | None = None
            for k in range(0, min(displacement_search, i) + 1):
                idx_k = i - k
                rng = highs[idx_k] - lows[idx_k]
                body = closes[idx_k] - opens[idx_k]
                a_k = atr_vals[idx_k]
                if not np.isnan(a_k) and rng > 0 and body >= a_k * min_displacement_atr and ((closes[idx_k] - lows[idx_k]) / rng) >= min_close_location:
                    disp_offset = k
                    break

            ob_offset: int | None = None
            if disp_offset is not None:
                for j in range(disp_offset + 1, min(ob_lookback, i) + 1):
                    idx_j = i - j
                    a_j = atr_vals[idx_j]
                    if not np.isnan(a_j) and (closes[idx_j] < opens[idx_j]) and ((highs[idx_j] - lows[idx_j]) >= a_j * min_ob_atr):
                        ob_offset = j
                        break

            if ob_offset is not None:
                bull_cand_high = float(highs[i - ob_offset])
                bull_cand_low = float(lows[i - ob_offset])
                bull_cand_bar = i - ob_offset
                bull_pending_since = i

                matched_fvg_offset: int | None = None
                matched_fvg_high = np.nan
                matched_fvg_low = np.nan

                for f_off in range(0, min(fvg_search_bars, i) + 1):
                    idx_f = i - f_off
                    if idx_f - 2 < 0 or f_off >= ob_offset:
                        continue
                    vol_idx = idx_f - 1
                    vol_pass = (not use_volume_filter) or (
                        vol_idx >= 0 and not np.isnan(avg_vol_vals[vol_idx]) and vols[vol_idx] >= avg_vol_vals[vol_idx] * min_fvg_volume_ratio
                    )
                    gap_low = float(highs[idx_f - 2])
                    gap_high = float(lows[idx_f])
                    a_f = atr_vals[idx_f]
                    if not np.isnan(a_f) and (gap_high > gap_low) and ((gap_high - gap_low) >= a_f * min_fvg_atr) and vol_pass:
                        if _intervals_overlap(bull_cand_low, bull_cand_high, gap_low, gap_high):
                            matched_fvg_offset = f_off
                            matched_fvg_high = gap_high
                            matched_fvg_low = gap_low
                            break

                if matched_fvg_offset is not None:
                    zones.append(
                        PairedZone(
                            direction=1,
                            ob_high=bull_cand_high,
                            ob_low=bull_cand_low,
                            fvg_high=float(matched_fvg_high),
                            fvg_low=float(matched_fvg_low),
                            ob_bar=bull_cand_bar,
                            fvg_bar=i - matched_fvg_offset,
                            created_bar=i,
                            trend_alignment=_align_label(1),
                        )
                    )
                    new_bull_setup = True
                else:
                    bull_pending = True

        # 11. Bearish BOS -> Displacement -> OB -> Overlapping FVG
        if bear_bos:
            bear_pending = False
            disp_offset = None
            for k in range(0, min(displacement_search, i) + 1):
                idx_k = i - k
                rng = highs[idx_k] - lows[idx_k]
                body = opens[idx_k] - closes[idx_k]
                a_k = atr_vals[idx_k]
                if not np.isnan(a_k) and rng > 0 and body >= a_k * min_displacement_atr and ((highs[idx_k] - closes[idx_k]) / rng) >= min_close_location:
                    disp_offset = k
                    break

            ob_offset = None
            if disp_offset is not None:
                for j in range(disp_offset + 1, min(ob_lookback, i) + 1):
                    idx_j = i - j
                    a_j = atr_vals[idx_j]
                    if not np.isnan(a_j) and (closes[idx_j] > opens[idx_j]) and ((highs[idx_j] - lows[idx_j]) >= a_j * min_ob_atr):
                        ob_offset = j
                        break

            if ob_offset is not None:
                bear_cand_high = float(highs[i - ob_offset])
                bear_cand_low = float(lows[i - ob_offset])
                bear_cand_bar = i - ob_offset
                bear_pending_since = i

                matched_fvg_offset = None
                matched_fvg_high = np.nan
                matched_fvg_low = np.nan

                for f_off in range(0, min(fvg_search_bars, i) + 1):
                    idx_f = i - f_off
                    if idx_f - 2 < 0 or f_off >= ob_offset:
                        continue
                    vol_idx = idx_f - 1
                    vol_pass = (not use_volume_filter) or (
                        vol_idx >= 0 and not np.isnan(avg_vol_vals[vol_idx]) and vols[vol_idx] >= avg_vol_vals[vol_idx] * min_fvg_volume_ratio
                    )
                    gap_high = float(lows[idx_f - 2])
                    gap_low = float(highs[idx_f])
                    a_f = atr_vals[idx_f]
                    if not np.isnan(a_f) and (gap_high > gap_low) and ((gap_high - gap_low) >= a_f * min_fvg_atr) and vol_pass:
                        if _intervals_overlap(bear_cand_low, bear_cand_high, gap_low, gap_high):
                            matched_fvg_offset = f_off
                            matched_fvg_high = gap_high
                            matched_fvg_low = gap_low
                            break

                if matched_fvg_offset is not None:
                    zones.append(
                        PairedZone(
                            direction=-1,
                            ob_high=bear_cand_high,
                            ob_low=bear_cand_low,
                            fvg_high=float(matched_fvg_high),
                            fvg_low=float(matched_fvg_low),
                            ob_bar=bear_cand_bar,
                            fvg_bar=i - matched_fvg_offset,
                            created_bar=i,
                            trend_alignment=_align_label(-1),
                        )
                    )
                    new_bear_setup = True
                else:
                    bear_pending = True

        # 12. Post-BOS pending FVG resolution on current completed bar
        vol_prev_ok = (not use_volume_filter) or (
            i - 1 >= 0 and not np.isnan(avg_vol_vals[i - 1]) and vols[i - 1] >= avg_vol_vals[i - 1] * min_fvg_volume_ratio
        )
        curr_bull_fvg = (i - 2 >= 0) and not np.isnan(atr_vals[i]) and vol_prev_ok and (l > highs[i - 2]) and ((l - highs[i - 2]) >= atr_vals[i] * min_fvg_atr)
        curr_bear_fvg = (i - 2 >= 0) and not np.isnan(atr_vals[i]) and vol_prev_ok and (h < lows[i - 2]) and ((lows[i - 2] - h) >= atr_vals[i] * min_fvg_atr)

        if bull_pending and bull_pending_since is not None and i > bull_pending_since:
            if i - bull_pending_since > fvg_search_bars:
                bull_pending = False
            elif curr_bull_fvg and _intervals_overlap(bull_cand_low, bull_cand_high, float(highs[i - 2]), float(l)):
                zones.append(
                    PairedZone(
                        direction=1,
                        ob_high=bull_cand_high,
                        ob_low=bull_cand_low,
                        fvg_high=float(l),
                        fvg_low=float(highs[i - 2]),
                        ob_bar=int(bull_cand_bar),
                        fvg_bar=i,
                        created_bar=i,
                        trend_alignment=_align_label(1),
                    )
                )
                new_bull_setup = True
                bull_pending = False

        if bear_pending and bear_pending_since is not None and i > bear_pending_since:
            if i - bear_pending_since > fvg_search_bars:
                bear_pending = False
            elif curr_bear_fvg and _intervals_overlap(bear_cand_low, bear_cand_high, float(h), float(lows[i - 2])):
                zones.append(
                    PairedZone(
                        direction=-1,
                        ob_high=bear_cand_high,
                        ob_low=bear_cand_low,
                        fvg_high=float(lows[i - 2]),
                        fvg_low=float(h),
                        ob_bar=int(bear_cand_bar),
                        fvg_bar=i,
                        created_bar=i,
                        trend_alignment=_align_label(-1),
                    )
                )
                new_bear_setup = True
                bear_pending = False

        # 13. Paired zone invalidation on close
        invalidated_count = 0
        for idx in range(len(zones) - 1, -1, -1):
            z = zones[idx]
            invalid = (c < z.ob_low or c < z.fvg_low) if z.direction == 1 else (c > z.ob_high or c > z.fvg_high)
            if invalid:
                zones.pop(idx)
                invalidated_count += 1

        # 14. Max historical setups cap
        while len(zones) > max_historical_setups:
            zones.pop(0)

        records.append(
            {
                "timestamp": data.index[i],
                "trend_direction": trend_dir,
                "bull_bos": bull_bos,
                "bear_bos": bear_bos,
                "new_bull_setup": new_bull_setup,
                "new_bear_setup": new_bear_setup,
                "swept_liquidity_count": swept_count,
                "invalidated_zones_count": invalidated_count,
                "active_zones_count": len(zones),
                "active_liquidity_count": len(liq_levels),
            }
        )

    return {
        "active_zones": zones,
        "active_liquidity": liq_levels,
        "signals_df": pd.DataFrame(records).set_index("timestamp"),
    }

