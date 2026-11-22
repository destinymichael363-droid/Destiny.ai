from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class SRZone:
    zone_type: Literal["resistance", "support"]  # 1 = resistance, -1 = support
    top: float
    bottom: float
    created_bar_index: int
    pivot_bar_index: int
    touch_count: int = 0
    state: int = 0  # 0 = waiting for first touch, 1 = inside zone, 2 = left zone, waiting for next visit
    is_strong: bool = False


def compute_atr(df: pd.DataFrame, atr_length: int = 14) -> pd.Series:
    """Wilder's ATR matching Pine Script's ta.atr(atrLength)."""
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

    # Wilder's RMA equivalent to ta.rma(ta.tr, atrLength)
    return tr.ewm(alpha=1.0 / atr_length, min_periods=atr_length, adjust=False).mean()


def find_pivots(series: pd.Series, left_len: int, right_len: int) -> pd.Series:
    """
    Non-repainting pivot detector matching ta.pivothigh / ta.pivotlow:
    A pivot at index `p` is confirmed at index `p + right_len` once `right_len`
    subsequent bars have formed. Returns the pivot value placed at the
    *confirmation* bar index so there is zero lookahead bias.
    """
    values = series.to_numpy(dtype=float)
    n = len(values)
    confirmed = np.full(n, np.nan, dtype=float)

    for p in range(left_len, n - right_len):
        center = values[p]
        if np.isnan(center):
            continue
        left_window = values[p - left_len : p]
        right_window = values[p + 1 : p + right_len + 1]

        if np.all(center > left_window) and np.all(center >= right_window):
            confirmed[p + right_len] = center

    return pd.Series(confirmed, index=series.index)


def run_sr_zones_engine(
    df: pd.DataFrame,
    length: int = 8,
    strength: int = 3,
    atr_length: int = 14,
    invalidation_mode: Literal["Wick", "Close"] = "Wick",
    max_zones: int = 40,
) -> dict:
    """
    Runs your 'Swing S/R Zones - destiny Style v2' logic bar-by-bar.
    Expects a DataFrame with columns: Open, High, Low, Close.
    """
    required = ["Open", "High", "Low", "Close"]
    if df is None or df.empty or not all(col in df.columns for col in required):
        return {
            "active_zones": [],
            "events_df": pd.DataFrame(),
        }

    data = df.copy()
    atr = compute_atr(data, atr_length=atr_length)
    pivot_highs = find_pivots(data["High"], length, length)
    pivot_lows = find_pivots(data["Low"], length, length)

    highs = data["High"].to_numpy(dtype=float)
    lows = data["Low"].to_numpy(dtype=float)
    closes = data["Close"].to_numpy(dtype=float)
    atr_vals = atr.to_numpy(dtype=float)
    ph_vals = pivot_highs.to_numpy(dtype=float)
    pl_vals = pivot_lows.to_numpy(dtype=float)

    zones: list[SRZone] = []
    bar_records: list[dict] = []

    for i in range(len(data)):
        h = highs[i]
        l = lows[i]
        c = closes[i]

        resistance_invalidated = False
        support_invalidated = False
        new_rejections = 0

        # 1. Create Resistance zone on confirmed pivot high
        atr_at_pivot = atr_vals[i - length] if i >= length else np.nan
        if not np.isnan(ph_vals[i]) and not np.isnan(atr_at_pivot):
            top = float(ph_vals[i])
            bottom = float(top - atr_at_pivot)
            zones.append(
                SRZone(
                    zone_type="resistance",
                    top=top,
                    bottom=bottom,
                    created_bar_index=i,
                    pivot_bar_index=i - length,
                    touch_count=0,
                    state=0,
                    is_strong=False,
                )
            )

        # 2. Create Support zone on confirmed pivot low
        if not np.isnan(pl_vals[i]) and not np.isnan(atr_at_pivot):
            bottom = float(pl_vals[i])
            top = float(bottom + atr_at_pivot)
            zones.append(
                SRZone(
                    zone_type="support",
                    top=top,
                    bottom=bottom,
                    created_bar_index=i,
                    pivot_bar_index=i - length,
                    touch_count=0,
                    state=0,
                    is_strong=False,
                )
            )

        # 3. Rejection counting and invalidation (reverse iteration like Pine loop)
        for idx in range(len(zones) - 1, -1, -1):
            z = zones[idx]

            if z.zone_type == "resistance":
                invalid = (h > z.top) if invalidation_mode == "Wick" else (c > z.top)
            else:
                invalid = (l < z.bottom) if invalidation_mode == "Wick" else (c < z.bottom)

            if invalid:
                if z.zone_type == "resistance":
                    resistance_invalidated = True
                else:
                    support_invalidated = True
                zones.pop(idx)
                continue

            inside = (h >= z.bottom) and (l <= z.top)
            rejected = (c < z.bottom) if z.zone_type == "resistance" else (c > z.top)

            if z.state == 0 and inside:
                z.state = 1

            if z.state == 1:
                left_zone = (h < z.bottom) if z.zone_type == "resistance" else (l > z.top)
                if left_zone:
                    z.state = 2

            if z.state == 2 and inside and rejected:
                z.touch_count += 1
                z.state = 1
                z.is_strong = z.touch_count >= strength
                new_rejections += 1

        # 4. Enforce maximum zone limit (drop oldest first, matching array.shift)
        while len(zones) > max_zones:
            zones.pop(0)

        bar_records.append(
            {
                "timestamp": data.index[i],
                "active_zone_count": len(zones),
                "resistance_invalidated": resistance_invalidated,
                "support_invalidated": support_invalidated,
                "new_rejections": new_rejections,
            }
        )

    events_df = pd.DataFrame(bar_records).set_index("timestamp")
    return {
        "active_zones": zones,
        "events_df": events_df,
    }
