from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class SRZone:
    zone_type: str      # "support" or "resistance"
    bottom: float
    top: float
    start_idx: int
    tf: str = ""
    strength: int = 1
    invalidated: bool = False
    end_idx: int | None = None


def _is_broken(
    zone_type: str,
    bottom: float,
    top: float,
    high_val: float,
    low_val: float,
    close_val: float,
    invalidation_mode: str,
) -> bool:
    mode = (invalidation_mode or "Wick").strip().lower()
    check_high = high_val if mode == "wick" else close_val
    check_low = low_val if mode == "wick" else close_val

    # Support breaks downward when price pierces/closes below bottom
    if zone_type == "support":
        return check_low < bottom
    # Resistance breaks upward when price pierces/closes above top
    return check_high > top


def run_sr_zones_engine(
    df: pd.DataFrame,
    tf_label: str = "15M",
    swing_length: int = 5,
    invalidation_mode: str = "Wick",
    max_active_zones: int = 4,
) -> dict:
    if df is None or df.empty or len(df) < (swing_length * 2 + 2):
        return {"active_zones": [], "minor_zones": [], "all_zones": []}

    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    n_bars = len(df)

    price_range = max(float(np.max(highs) - np.min(lows)), 1e-6)
    pad = price_range * 0.006

    candidates: list[SRZone] = []

    for i in range(swing_length, n_bars - swing_length):
        if highs[i] == np.max(highs[i - swing_length : i + swing_length + 1]):
            px = float(highs[i])
            candidates.append(
                SRZone(
                    zone_type="resistance",
                    bottom=px - pad,
                    top=px + pad,
                    start_idx=i,
                    tf=tf_label,
                    strength=1,
                )
            )
        if lows[i] == np.min(lows[i - swing_length : i + swing_length + 1]):
            px = float(lows[i])
            candidates.append(
                SRZone(
                    zone_type="support",
                    bottom=px - pad,
                    top=px + pad,
                    start_idx=i,
                    tf=tf_label,
                    strength=1,
                )
            )

    evaluated: list[SRZone] = []
    for zone in candidates:
        broken_at: int | None = None
        for j in range(zone.start_idx + 1, n_bars):
            if _is_broken(
                zone_type=zone.zone_type,
                bottom=zone.bottom,
                top=zone.top,
                high_val=float(highs[j]),
                low_val=float(lows[j]),
                close_val=float(closes[j]),
                invalidation_mode=invalidation_mode,
            ):
                broken_at = j
                break

        zone.invalidated = broken_at is not None
        zone.end_idx = broken_at if broken_at is not None else (n_bars - 1)
        evaluated.append(zone)

    active_zones = [z for z in evaluated if not z.invalidated][-max_active_zones:]
    minor_zones = [z for z in evaluated if not z.invalidated][-(max_active_zones * 2) : -max_active_zones]

    return {
        "active_zones": active_zones,
        "minor_zones": minor_zones,
        "all_zones": evaluated,
    }
