"""
DESTINY AI — SUPPORT & RESISTANCE ENGINE
Version: 4.0

Purpose
-------
Historical chart analysis and virtual-paper-testing support only.
This module does not execute trades or connect to a broker.

Design
------
- Finds confirmed swing highs and lows using a symmetric pivot window.
- A pivot is not available until `pivot_length` completed candles have
  formed to its right.
- Builds support/resistance zones from ATR measured at the pivot candle.
- Replays candles chronologically so a zone cannot react to candles that
  occurred before the zone was confirmed.
- Uses completed-candle CLOSE invalidation by default.
- Counts a rejection only after an initial visit, a departure, and a
  subsequent visit that closes on the expected side of the zone.
- Keeps the public function names and main output columns compatible
  with the previous Destiny AI S/R engine.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd


InvalidationMode = Literal["Wick", "Close"]


OUTPUT_COLUMNS = [
    "Side",
    "Zone Low",
    "Zone High",
    "Confirmed Time",
    "Pivot Time",
    "Rejections",
    "Strength",
    "Midline",
    "Strong",
]


def _validate_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return a clean, chronological OHLC frame or raise a useful error."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("S/R input must be a pandas DataFrame.")

    required = ["High", "Low", "Close"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(
            "S/R input is missing required columns: " + ", ".join(missing)
        )

    clean = data.copy()
    for column in required:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=required)
    clean = clean.sort_index()
    clean = clean[~clean.index.duplicated(keep="last")]

    if clean.empty:
        raise ValueError("S/R input has no valid OHLC candles.")

    invalid_range = clean["High"] < clean["Low"]
    if invalid_range.any():
        raise ValueError("S/R input contains candles where High is below Low.")

    return clean


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Wilder-style ATR using only the supplied candle history."""
    if period < 1:
        raise ValueError("ATR period must be at least 1.")

    clean = _validate_data(data)
    previous_close = clean["Close"].shift(1)

    true_range = pd.concat(
        [
            clean["High"] - clean["Low"],
            (clean["High"] - previous_close).abs(),
            (clean["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder smoothing; min_periods keeps early, under-sampled ATR values NaN.
    atr = true_range.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    # A usable fallback for short histories, without fabricating price data.
    fallback = true_range.expanding(min_periods=1).mean()
    return atr.fillna(fallback).rename("ATR")


def find_confirmed_pivots(
    data: pd.DataFrame,
    length: int = 8,
) -> pd.DataFrame:
    """
    Find confirmed swing highs/lows.

    The returned row is timestamped at the candle where the pivot becomes
    confirmed, while Pivot Time identifies the original swing candle.
    Equal highs/lows are handled deterministically: the first occurrence
    in the pivot window is retained.
    """
    if length < 1:
        raise ValueError("Pivot length must be at least 1.")

    clean = _validate_data(data)
    atr = calculate_atr(clean, period=14)

    records = []
    highs = clean["High"].to_numpy(dtype=float)
    lows = clean["Low"].to_numpy(dtype=float)
    index = clean.index

    # i is the pivot candle; confirmation occurs `length` bars later.
    for i in range(length, len(clean) - length):
        left = i - length
        right = i + length + 1

        high_window = highs[left:right]
        low_window = lows[left:right]

        is_pivot_high = (
            highs[i] == np.max(high_window)
            and np.argmax(high_window) == length
        )
        is_pivot_low = (
            lows[i] == np.min(low_window)
            and np.argmin(low_window) == length
        )

        confirmation_position = i + length
        if is_pivot_high:
            records.append(
                {
                    "Position": confirmation_position,
                    "Type": "RESISTANCE",
                    "Price": float(highs[i]),
                    "Pivot Time": index[i],
                    "Confirmed Time": index[confirmation_position],
                    "ATR": float(atr.iloc[i]),
                }
            )

        if is_pivot_low:
            records.append(
                {
                    "Position": confirmation_position,
                    "Type": "SUPPORT",
                    "Price": float(lows[i]),
                    "Pivot Time": index[i],
                    "Confirmed Time": index[confirmation_position],
                    "ATR": float(atr.iloc[i]),
                }
            )

    columns = [
        "Position",
        "Type",
        "Price",
        "Pivot Time",
        "Confirmed Time",
        "ATR",
    ]
    return pd.DataFrame(records, columns=columns)


def _zone_bounds(
    side: str,
    pivot_price: float,
    atr_at_pivot: float,
    zone_atr: float,
) -> tuple[float, float]:
    """
    Build a one-sided zone around the swing point.

    Support extends below its swing low; resistance extends above its
    swing high. Zone width is based on ATR at the original pivot candle.
    """
    safe_atr = atr_at_pivot if np.isfinite(atr_at_pivot) and atr_at_pivot > 0 else 0.0
    width = safe_atr * zone_atr

    if side == "SUPPORT":
        return pivot_price - width, pivot_price
    return pivot_price, pivot_price + width


def analyze_sr(
    data: pd.DataFrame,
    pivot_length: int = 8,
    zone_atr: float = 1.0,
    reaction_atr: float = 0.5,
    atr_length: int = 14,
    strength: int = 3,
    invalid_mode: InvalidationMode = "Close",
    max_zones: int = 40,
) -> pd.DataFrame:
    """
    Analyse chronological support/resistance zones.

    Parameters retained for compatibility:
      pivot_length: left/right bars required to confirm a swing.
      zone_atr: zone width as a multiple of ATR at the pivot candle.
      reaction_atr: reserved for future reaction-distance classification.
      atr_length: ATR calculation period.
      strength: rejection count needed for Strong=True.
      invalid_mode: "Close" (recommended) or "Wick".
      max_zones: maximum number of most recently confirmed zones returned.

    Output columns remain compatible with the previous Destiny AI app.
    """
    del reaction_atr  # Kept in the public signature for compatibility.

    if pivot_length < 1:
        raise ValueError("pivot_length must be at least 1.")
    if zone_atr < 0:
        raise ValueError("zone_atr cannot be negative.")
    if atr_length < 1:
        raise ValueError("atr_length must be at least 1.")
    if strength < 1:
        raise ValueError("strength must be at least 1.")
    if max_zones < 1:
        raise ValueError("max_zones must be at least 1.")
    if invalid_mode not in ("Wick", "Close"):
        raise ValueError("invalid_mode must be either 'Wick' or 'Close'.")

    clean = _validate_data(data)
    if len(clean) < (pivot_length * 2 + 1):
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    atr = calculate_atr(clean, period=atr_length)
    pivots = find_confirmed_pivots(clean, length=pivot_length)

    if pivots.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Group pivots by confirmation bar, preserving deterministic order.
    pivots_by_position = {}
    for pivot in pivots.to_dict("records"):
        pivots_by_position.setdefault(int(pivot["Position"]), []).append(pivot)

    active_zones = []
    completed_zones = []

    highs = clean["High"].to_numpy(dtype=float)
    lows = clean["Low"].to_numpy(dtype=float)
    closes = clean["Close"].to_numpy(dtype=float)
    times = clean.index

    for position in range(len(clean)):
        high = highs[position]
        low = lows[position]
        close = closes[position]
        current_time = times[position]

        # First update zones that were already known before this candle.
        survivors = []
        for zone in active_zones:
            side = zone["Side"]

            # Invalidation is evaluated before touch/rejection counting.
            if invalid_mode == "Close":
                invalid = (
                    close < zone["Zone Low"]
                    if side == "SUPPORT"
                    else close > zone["Zone High"]
                )
            else:
                invalid = (
                    low < zone["Zone Low"]
                    if side == "SUPPORT"
                    else high > zone["Zone High"]
                )

            if invalid:
                zone["Invalidated Time"] = current_time
                continue

            intersects = high >= zone["Zone Low"] and low <= zone["Zone High"]

            # 0 = waiting for first visit
            # 1 = visited, waiting for price to leave
            # 2 = left after a visit, waiting for a later revisit
            if intersects:
                favorable_close = (
                    close >= zone["Midline"]
                    if side == "SUPPORT"
                    else close <= zone["Midline"]
                )

                if zone["_visit_state"] == 2 and favorable_close:
                    zone["Rejections"] += 1
                    zone["_visit_state"] = 1
                elif zone["_visit_state"] == 0:
                    zone["_visit_state"] = 1
                elif zone["_visit_state"] == 2:
                    # A revisit that does not close favorably is not counted.
                    zone["_visit_state"] = 1
            elif zone["_visit_state"] == 1:
                zone["_visit_state"] = 2

            zone["Strength"] = min(int(zone["Rejections"]), strength)
            zone["Strong"] = int(zone["Rejections"]) >= strength
            survivors.append(zone)

        active_zones = survivors

        # Add pivots only after existing zones have processed this candle.
        # This prevents a newly confirmed zone reacting to its own
        # confirmation candle.
        for pivot in pivots_by_position.get(position, []):
            side = pivot["Type"]
            zone_low, zone_high = _zone_bounds(
                side=side,
                pivot_price=float(pivot["Price"]),
                atr_at_pivot=float(pivot["ATR"]),
                zone_atr=float(zone_atr),
            )

            if not np.isfinite(zone_low) or not np.isfinite(zone_high):
                continue
            if zone_high <= zone_low:
                # If ATR is unavailable/zero, use a zero-width pivot level.
                zone_low = float(pivot["Price"])
                zone_high = float(pivot["Price"])

            midline = (zone_low + zone_high) / 2.0

            # Avoid adding nearly identical same-side zones.
            duplicate = False
            for existing in active_zones:
                if existing["Side"] != side:
                    continue
                overlap = (
                    min(existing["Zone High"], zone_high)
                    - max(existing["Zone Low"], zone_low)
                )
                smaller_width = min(
                    max(existing["Zone High"] - existing["Zone Low"], 1e-12),
                    max(zone_high - zone_low, 1e-12),
                )
                if overlap > 0 and overlap / smaller_width >= 0.70:
                    duplicate = True
                    break

            if duplicate:
                continue

            zone = {
                "Side": side,
                "Zone Low": float(zone_low),
                "Zone High": float(zone_high),
                "Confirmed Time": pivot["Confirmed Time"],
                "Pivot Time": pivot["Pivot Time"],
                "Rejections": 0,
                "Strength": 0,
                "Midline": float(midline),
                "Strong": False,
                "_visit_state": 0,
            }
            active_zones.append(zone)

    # Return active zones only. Expired/invalidated zones are intentionally
    # excluded from the current chart/table.
    active_zones.sort(
        key=lambda item: item["Confirmed Time"],
        reverse=True,
    )
    output = []
    for zone in active_zones[:max_zones]:
        output.append({column: zone[column] for column in OUTPUT_COLUMNS})

    return pd.DataFrame(output, columns=OUTPUT_COLUMNS)


def get_sr_summary(zones: Optional[pd.DataFrame]) -> dict:
    """Small diagnostic summary for the app's S/R status panel."""
    if zones is None or not isinstance(zones, pd.DataFrame) or zones.empty:
        return {
            "total": 0,
            "support": 0,
            "resistance": 0,
            "strong": 0,
        }

    return {
        "total": int(len(zones)),
        "support": int((zones["Side"] == "SUPPORT").sum()),
        "resistance": int((zones["Side"] == "RESISTANCE").sum()),
        "strong": int(zones["Strong"].fillna(False).astype(bool).sum()),
    }
