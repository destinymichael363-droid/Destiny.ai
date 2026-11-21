# =========================================================
# DESTINY AI — SWING S/R ZONES (DESTINY STYLE V2)
# Educational chart-visualization engine; no trading signals.
# =========================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


InvalidationMode = Literal["Wick", "Close"]


@dataclass
class SRZone:
    side: str                 # "Support" or "Resistance"
    low: float
    high: float
    pivot_time: object
    confirmed_time: object
    rejections: int = 0
    state: int = 0           # 0 waiting touch, 1 touched, 2 left zone



_REQUIRED = {"High", "Low", "Close"}


def _validate_data(data: pd.DataFrame) -> pd.DataFrame:
    missing = _REQUIRED.difference(data.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {', '.join(sorted(missing))}")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    frame = data.copy()
    for col in ("High", "Low", "Close"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["High", "Low", "Close"])
    if (frame["High"] < frame["Low"]).any():
        raise ValueError("Candle High cannot be below Low")
    return frame


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder-style ATR (RMA), aligned to the input index."""
    if period < 1:
        raise ValueError("period must be >= 1")
    frame = _validate_data(data)
    previous_close = frame["Close"].shift(1)
    true_range = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - previous_close).abs(),
            (frame["Low"] - previous_close).abs(),
        ], axis=1,
    ).max(axis=1)
    # Pine ta.atr() uses Wilder/RMA smoothing. Seed with an initial SMA,
    # then recursively apply alpha=1/period.
    atr = pd.Series(float("nan"), index=frame.index, dtype=float)
    if len(true_range) >= period:
        seed_pos = period - 1
        atr.iloc[seed_pos] = true_range.iloc[:period].mean()
        for i in range(seed_pos + 1, len(true_range)):
            atr.iloc[i] = (atr.iloc[i - 1] * (period - 1) + true_range.iloc[i]) / period
    return atr


def find_confirmed_pivots(data: pd.DataFrame, length: int = 8) -> list[dict]:
    """Return pivot records, with confirmation occurring length bars after pivot."""
    if length < 2:
        raise ValueError("length must be >= 2")
    frame = _validate_data(data)
    highs = frame["High"].to_numpy()
    lows = frame["Low"].to_numpy()
    records: list[dict] = []

    for confirm_i in range(length * 2, len(frame)):
        pivot_i = confirm_i - length
        left = pivot_i - length
        right = confirm_i
        high_window = highs[left : right + 1]
        low_window = lows[left : right + 1]
        # Use strict unique extrema to avoid duplicate pivots on flat plateaus.
        if len(high_window) and highs[pivot_i] == high_window.max() and (high_window == highs[pivot_i]).sum() == 1:
            records.append({"side": "Resistance", "pivot_i": pivot_i, "confirm_i": confirm_i, "price": float(highs[pivot_i])})
        if len(low_window) and lows[pivot_i] == low_window.min() and (low_window == lows[pivot_i]).sum() == 1:
            records.append({"side": "Support", "pivot_i": pivot_i, "confirm_i": confirm_i, "price": float(lows[pivot_i])})
    return records



def analyze_sr(
    data: pd.DataFrame,
    pivot_length: int = 8,
    zone_atr: float = 1.0,
    reaction_atr: float = 0.5,  # retained for compatibility; not used by this indicator
    atr_length: int = 14,
    strength: int = 3,
    invalid_mode: InvalidationMode = "Wick",
    max_zones: int = 40,
) -> pd.DataFrame:
    """Replay S/R zone state chronologically and return currently active zones.



    Zone width is one ATR at the pivot candle, matching the supplied Pine script.
    ``zone_atr`` remains in the signature for older callers but is intentionally
    ignored. Rejection counting follows the Pine script's leave-and-revisit state.
    """
    frame = _validate_data(data)
    if pivot_length < 2 or atr_length < 1 or strength < 1 or max_zones < 1:
        raise ValueError("Invalid S/R settings")
    if invalid_mode not in ("Wick", "Close"):
        raise ValueError("invalid_mode must be 'Wick' or 'Close'")

    atr = calculate_atr(frame, atr_length)
    pivots_by_confirm: dict[int, list[dict]] = {}
    for pivot in find_confirmed_pivots(frame, pivot_length):
        pivots_by_confirm.setdefault(pivot["confirm_i"], []).append(pivot)


    active: list[SRZone] = []
    for i in range(len(frame)):
        row = frame.iloc[i]
        high, low, close = float(row["High"]), float(row["Low"]), float(row["Close"])


        # Create zones only when the pivot is confirmed; zone width uses ATR at pivot.
        for p in pivots_by_confirm.get(i, []):
            width = atr.iloc[p["pivot_i"]]
            if pd.isna(width) or width <= 0:
                continue
            if p["side"] == "Resistance":
                top, bottom = p["price"], p["price"] - float(width)
            else:
                bottom, top = p["price"], p["price"] + float(width)
            active.append(SRZone(
                side=p["side"], low=float(bottom), high=float(top),
                pivot_time=frame.index[p["pivot_i"]], confirmed_time=frame.index[i],
            ))
            while len(active) > max_zones:
                active.pop(0)

        # Update every zone using the current candle, oldest-to-newest.
        survivors: list[SRZone] = []
        for zone in active:
            if zone.side == "Resistance":
                invalid = high > zone.high if invalid_mode == "Wick" else close > zone.high
                inside = high >= zone.low and low <= zone.high
                rejected = close < zone.low
                left_zone = high < zone.low
            else:
                invalid = low < zone.low if invalid_mode == "Wick" else close < zone.low
                inside = high >= zone.low and low <= zone.high
                rejected = close > zone.high
                left_zone = low > zone.high

            if invalid:
                continue

            # Match the original sequential Pine state transitions.
            if zone.state == 0 and inside:
                zone.state = 1
            elif zone.state == 1 and left_zone:
                zone.state = 2
            elif zone.state == 2 and inside and rejected:
                zone.rejections += 1
                zone.state = 1
            survivors.append(zone)
        active = survivors


    result = []
    for zone in active:
        capped_strength = min(zone.rejections, strength)
        result.append({
            "Side": zone.side,
            "Zone Low": zone.low,
            "Zone High": zone.high,
            "Confirmed Time": zone.confirmed_time,
            "Pivot Time": zone.pivot_time,
            "Rejections": zone.rejections,
            "Strength": capped_strength,
            "Midline": (zone.low + zone.high) / 2.0,
            "Strong": zone.rejections >= strength,
        })
    out = pd.DataFrame(result)
    if out.empty:
        return pd.DataFrame(columns=["Side", "Zone Low", "Zone High", "Confirmed Time", "Pivot Time", "Rejections", "Strength", "Midline", "Strong"])
    return out.sort_values("Confirmed Time", ascending=False).reset_index(drop=True)
