from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class SRZone:
    zone_type: str  # "support" / "resistance"
    bottom: float
    top: float
    start_idx: int
    tf: str = ""
    strength: int = 1
    invalidated: bool = False
    end_idx: int | None = None

    # New metadata.
    # start_idx remains the original pivot location so existing callers
    # do not lose that information.
    confirmed_idx: int | None = None


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Wilder-style ATR.

    Uses only information available up to each candle.
    """
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / max(int(length), 1),
        adjust=False,
        min_periods=1,
    ).mean()


def _is_broken(
    zone_type: str,
    bottom: float,
    top: float,
    high_val: float,
    low_val: float,
    close_val: float,
    invalidation_mode: str,
) -> bool:
    """
    Determine whether a zone has genuinely broken.

    Close-based invalidation is generally safer for S/R because a wick
    through a level can represent a liquidity sweep rather than a true
    structural break.

    Supported modes:
        - "close"
        - "wick"
    """
    mode = str(invalidation_mode or "close").strip().lower()

    if mode == "wick":
        check_high = high_val
        check_low = low_val
    else:
        # Close mode deliberately ignores temporary wick penetration.
        check_high = close_val
        check_low = close_val

    if zone_type == "support":
        return check_low < bottom

    if zone_type == "resistance":
        return check_high > top

    return False


def _valid_ohlc(df: pd.DataFrame) -> bool:
    required = {"High", "Low", "Close"}
    return required.issubset(df.columns)


def run_sr_zones_engine(
    df: pd.DataFrame,
    tf_label: str = "15M",
    swing_length: int = 5,
    invalidation_mode: Literal["Wick", "Close"] = "Close",
    max_active_zones: int = 4,
    atr_length: int = 14,
    zone_atr_mult: float = 0.20,
    min_zone_width: float = 0.0,
):
    """
    Detect confirmed support/resistance zones without introducing
    future-data lookahead.

    IMPORTANT DESIGN CHANGE
    -----------------------
    A swing at index i requires `swing_length` candles to the right
    before it is confirmed.

    Therefore:

        pivot_idx = i
        confirmed_idx = i + swing_length

    The pivot's price is still taken from the original pivot candle,
    but the zone is NOT considered available until confirmed_idx.

    This distinction is critical for realistic backtesting and live
    execution.

    Returns the same overall structure as the previous engine:

        {
            "active_zones": [...],
            "minor_zones": [...],
            "all_zones": [...]
        }
    """

    # ------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------
    if df is None or not isinstance(df, pd.DataFrame):
        return {
            "active_zones": [],
            "minor_zones": [],
            "all_zones": [],
        }

    if not _valid_ohlc(df):
        return {
            "active_zones": [],
            "minor_zones": [],
            "all_zones": [],
        }

    try:
        swing_length = max(int(swing_length), 1)
    except Exception:
        swing_length = 5

    try:
        max_active_zones = max(int(max_active_zones), 1)
    except Exception:
        max_active_zones = 4

    try:
        atr_length = max(int(atr_length), 1)
    except Exception:
        atr_length = 14

    try:
        zone_atr_mult = max(float(zone_atr_mult), 0.0)
    except Exception:
        zone_atr_mult = 0.20

    try:
        min_zone_width = max(float(min_zone_width), 0.0)
    except Exception:
        min_zone_width = 0.0

    data = df.copy()

    # Numeric normalization.
    for column in ("High", "Low", "Close"):
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(subset=["High", "Low", "Close"])

    if len(data) < swing_length * 2 + 2:
        return {
            "active_zones": [],
            "minor_zones": [],
            "all_zones": [],
        }

    data = data.reset_index(drop=False)

    highs = data["High"].to_numpy(dtype=float)
    lows = data["Low"].to_numpy(dtype=float)
    closes = data["Close"].to_numpy(dtype=float)

    n = len(data)

    # ATR is calculated sequentially. For a pivot we use ATR available
    # at the moment the pivot becomes confirmed.
    atr_series = _atr(data, atr_length)

    zones: list[SRZone] = []

    # ------------------------------------------------------------
    # 1. Detect raw swing candidates
    # ------------------------------------------------------------
    #
    # A pivot at i is only known after candles:
    #
    #     i+1 ... i+swing_length
    #
    # have closed.
    #
    # This is the most important lookahead correction.
    # ------------------------------------------------------------

    for i in range(
        swing_length,
        n - swing_length,
    ):
        left = i - swing_length
        right = i + swing_length + 1

        high_window = highs[left:right]
        low_window = lows[left:right]

        pivot_high = highs[i] >= np.max(high_window)
        pivot_low = lows[i] <= np.min(low_window)

        confirmed_idx = i + swing_length

        if confirmed_idx >= n:
            continue

        # ATR known at the confirmation candle.
        atr_value = float(atr_series.iloc[confirmed_idx])

        if not np.isfinite(atr_value) or atr_value <= 0:
            # Fallback to local range if ATR is unavailable.
            local_range = highs[left:right].max() - lows[left:right].min()
            atr_value = float(local_range)

        if not np.isfinite(atr_value) or atr_value <= 0:
            continue

        # Adaptive zone half-width.
        #
        # FIX: `min_zone_width` was previously used as a HALF-pad
        # (added to each side of the pivot), even though its name
        # implies "minimum total zone width". Anyone setting
        # min_zone_width=0.5 on EURUSD (~1.08) would have produced
        # a 1.0-wide zone — the entire price range.
        #
        # Now it is interpreted as the minimum TOTAL width, so the
        # half-pad is min_zone_width / 2. The default of 0.0 keeps
        # behavior identical for all existing callers.
        pad = max(
            atr_value * zone_atr_mult,
            min_zone_width / 2.0,
        )

        # --------------------------------------------------------
        # Resistance
        # --------------------------------------------------------
        if pivot_high:
            px = highs[i]

            zone = SRZone(
                zone_type="resistance",
                bottom=float(px - pad),
                top=float(px + pad),
                start_idx=i,
                tf=tf_label,
                strength=1,
                invalidated=False,
                end_idx=None,
                confirmed_idx=confirmed_idx,
            )

            zones.append(zone)

        # --------------------------------------------------------
        # Support
        # --------------------------------------------------------
        if pivot_low:
            px = lows[i]

            zone = SRZone(
                zone_type="support",
                bottom=float(px - pad),
                top=float(px + pad),
                start_idx=i,
                tf=tf_label,
                strength=1,
                invalidated=False,
                end_idx=None,
                confirmed_idx=confirmed_idx,
            )

            zones.append(zone)

    # ------------------------------------------------------------
    # 2. Sort chronologically
    # ------------------------------------------------------------
    #
    # The original pivot location is retained in start_idx.
    # confirmed_idx determines when the zone becomes usable.
    #
    # FIX: previous sort key used `z.confirmed_idx or z.start_idx`.
    # When confirmed_idx == 0 (valid early pivot), the `or` fell
    # through to start_idx, mis-sorting that zone. Now we check
    # `is not None` explicitly.
    # ------------------------------------------------------------

    zones.sort(
        key=lambda z: (
            z.confirmed_idx
            if z.confirmed_idx is not None
            else z.start_idx,
            z.start_idx,
        )
    )

    # ------------------------------------------------------------
    # 3. Evaluate each zone only AFTER confirmation
    # ------------------------------------------------------------
    #
    # A zone cannot be invalidated before it exists.
    #
    # This prevents historical bars between the pivot and its
    # confirmation from contaminating the result.
    # ------------------------------------------------------------

    for zone in zones:
        confirmation_idx = (
            zone.confirmed_idx
            if zone.confirmed_idx is not None
            else zone.start_idx
        )

        broken_at: int | None = None

        # Start at the confirmation candle.
        for j in range(
            max(confirmation_idx, 0),
            n,
        ):
            high_val = highs[j]
            low_val = lows[j]
            close_val = closes[j]

            if _is_broken(
                zone_type=zone.zone_type,
                bottom=zone.bottom,
                top=zone.top,
                high_val=high_val,
                low_val=low_val,
                close_val=close_val,
                invalidation_mode=invalidation_mode,
            ):
                broken_at = j
                break

        if broken_at is not None:
            zone.invalidated = True
            zone.end_idx = broken_at

    # ------------------------------------------------------------
    # 4. Determine currently active zones
    # ------------------------------------------------------------

    active = [
        zone
        for zone in zones
        if not zone.invalidated
    ]

    # Keep the newest active zones.
    active = active[-max_active_zones:]

    # ------------------------------------------------------------
    # 5. Minor zones
    # ------------------------------------------------------------
    #
    # Preserve the old concept of keeping a secondary historical
    # group immediately behind the active zones.
    # ------------------------------------------------------------

    active_ids = {id(zone) for zone in active}

    historical_active_candidates = [
        zone
        for zone in zones
        if not zone.invalidated
        and id(zone) not in active_ids
    ]

    minor_count = max_active_zones

    minor_zones = historical_active_candidates[-minor_count:]

    # ------------------------------------------------------------
    # 6. Return compatible structure
    # ------------------------------------------------------------

    return {
        "active_zones": active,
        "minor_zones": minor_zones,
        "all_zones": zones,
    }
