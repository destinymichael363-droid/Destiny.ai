
# =========================================================
# DESTINY AI — SWING S/R ZONES
# CLEAN CHRONOLOGICAL ENGINE V3
# Educational chart visualization only.
# =========================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


InvalidationMode = Literal["Wick", "Close"]

_REQUIRED = {"High", "Low", "Close"}


# =========================================================
# 1. S/R ZONE MODEL
# =========================================================

@dataclass
class SRZone:
    side: str
    low: float
    high: float
    pivot_time: object
    confirmed_time: object
    rejections: int = 0

    # 0 = waiting for first touch
    # 1 = touched, waiting to leave
    # 2 = left zone, waiting for a new visit
    state: int = 0


# =========================================================
# 2. DATA VALIDATION
# =========================================================

def _validate_data(data: pd.DataFrame) -> pd.DataFrame:

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    missing = _REQUIRED.difference(data.columns)

    if missing:
        raise ValueError(
            f"Missing candle columns: {', '.join(sorted(missing))}"
        )

    frame = data.copy()

    for col in ("High", "Low", "Close"):
        frame[col] = pd.to_numeric(
            frame[col],
            errors="coerce"
        )

    frame = frame.dropna(
        subset=["High", "Low", "Close"]
    )

    if (frame["High"] < frame["Low"]).any():
        raise ValueError(
            "Candle High cannot be below Low"
        )

    if frame.empty:
        raise ValueError("No valid candle data supplied")

    return frame


# =========================================================
# 3. ATR CALCULATION
# =========================================================

def calculate_atr(
    data: pd.DataFrame,
    period: int = 14
) -> pd.Series:

    if period < 1:
        raise ValueError("period must be >= 1")

    frame = _validate_data(data)

    previous_close = frame["Close"].shift(1)

    true_range = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - previous_close).abs(),
            (frame["Low"] - previous_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    atr = pd.Series(
        float("nan"),
        index=frame.index,
        dtype=float
    )

    if len(true_range) >= period:

        seed_pos = period - 1

        atr.iloc[seed_pos] = (
            true_range.iloc[:period].mean()
        )

        for i in range(seed_pos + 1, len(true_range)):

            atr.iloc[i] = (
                atr.iloc[i - 1] * (period - 1)
                + true_range.iloc[i]
            ) / period

    return atr


# =========================================================
# 4. CONFIRMED SWING PIVOTS
# =========================================================

def find_confirmed_pivots(
    data: pd.DataFrame,
    length: int = 8
) -> list[dict]:

    if length < 2:
        raise ValueError("length must be >= 2")

    frame = _validate_data(data)

    highs = frame["High"].to_numpy()
    lows = frame["Low"].to_numpy()

    records = []

    for confirm_i in range(length * 2, len(frame)):

        pivot_i = confirm_i - length

        left = pivot_i - length
        right = confirm_i

        high_window = highs[left:right + 1]
        low_window = lows[left:right + 1]

        # Unique swing high.
        if (
            len(high_window)
            and highs[pivot_i] == high_window.max()
            and (high_window == highs[pivot_i]).sum() == 1
        ):

            records.append({
                "side": "Resistance",
                "pivot_i": pivot_i,
                "confirm_i": confirm_i,
                "price": float(highs[pivot_i])
            })

        # Unique swing low.
        if (
            len(low_window)
            and lows[pivot_i] == low_window.min()
            and (low_window == lows[pivot_i]).sum() == 1
        ):

            records.append({
                "side": "Support",
                "pivot_i": pivot_i,
                "confirm_i": confirm_i,
                "price": float(lows[pivot_i])
            })

    return records


# =========================================================
# 5. MAIN CHRONOLOGICAL S/R ENGINE
# =========================================================

def analyze_sr(
    data: pd.DataFrame,
    pivot_length: int = 8,
    zone_atr: float = 1.0,
    reaction_atr: float = 0.5,
    atr_length: int = 14,
    strength: int = 3,
    invalid_mode: InvalidationMode = "Wick",
    max_zones: int = 40,
) -> pd.DataFrame:

    """
    Chronological support and resistance zone replay.

    A zone is created only after pivot confirmation.

    IMPORTANT:
    A newly confirmed zone is not processed against
    its confirmation candle. Processing starts on
    the following candle.

    Zone width uses ATR at the original pivot candle.

    zone_atr and reaction_atr remain in the signature
    for compatibility with existing callers.
    """

    frame = _validate_data(data)

    if (
        pivot_length < 2
        or atr_length < 1
        or strength < 1
        or max_zones < 1
    ):
        raise ValueError("Invalid S/R settings")

    if invalid_mode not in ("Wick", "Close"):
        raise ValueError(
            "invalid_mode must be 'Wick' or 'Close'"
        )

    # Calculate ATR.
    atr = calculate_atr(
        frame,
        atr_length
    )

    # Organize pivots by their confirmation candle.
    pivots_by_confirm = {}

    pivots = find_confirmed_pivots(
        frame,
        pivot_length
    )

    for pivot in pivots:

        pivots_by_confirm.setdefault(
            pivot["confirm_i"],
            []
        ).append(pivot)

    active: list[SRZone] = []

    # =====================================================
    # CHRONOLOGICAL REPLAY
    # =====================================================

    for i in range(len(frame)):

        row = frame.iloc[i]

        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])

        # -------------------------------------------------
        # A. PROCESS EXISTING ZONES ONLY
        # -------------------------------------------------

        survivors = []

        for zone in active:

            if zone.side == "Resistance":

                if invalid_mode == "Wick":
                    invalid = high > zone.high
                else:
                    invalid = close > zone.high

                inside = (
                    high >= zone.low
                    and low <= zone.high
                )

                rejected = close < zone.low

                left_zone = high < zone.low

            else:

                if invalid_mode == "Wick":
                    invalid = low < zone.low
                else:
                    invalid = close < zone.low

                inside = (
                    high >= zone.low
                    and low <= zone.high
                )

                rejected = close > zone.high

                left_zone = low > zone.high

            # Remove invalidated zones.
            if invalid:
                continue

            # Sequential visit state machine.
            if zone.state == 0 and inside:

                zone.state = 1

            elif zone.state == 1 and left_zone:

                zone.state = 2

            elif (
                zone.state == 2
                and inside
                and rejected
            ):

                zone.rejections += 1
                zone.state = 1

            survivors.append(zone)

        active = survivors

        # -------------------------------------------------
        # B. CREATE NEWLY CONFIRMED ZONES
        # -------------------------------------------------
        #
        # This happens AFTER existing-zone processing.
        # Therefore, a new zone cannot be touched,
        # rejected or invalidated on its creation candle.
        #
        # Its first eligible processing candle is i + 1.
        # -------------------------------------------------

        for pivot in pivots_by_confirm.get(i, []):

            pivot_i = pivot["pivot_i"]

            width = atr.iloc[pivot_i]

            if pd.isna(width) or width <= 0:
                continue

            price = float(pivot["price"])
            width = float(width)

            if pivot["side"] == "Resistance":

                top = price
                bottom = price - width

            else:

                bottom = price
                top = price + width

            new_zone = SRZone(
                side=pivot["side"],
                low=bottom,
                high=top,
                pivot_time=frame.index[pivot_i],
                confirmed_time=frame.index[i]
            )

            active.append(new_zone)

            # Keep the most recent zones.
            while len(active) > max_zones:
                active.pop(0)

    # =====================================================
    # 6. BUILD OUTPUT
    # =====================================================

    result = []

    for zone in active:

        capped_strength = min(
            zone.rejections,
            strength
        )

        result.append({

            "Side": zone.side,

            "Zone Low": zone.low,

            "Zone High": zone.high,

            "Confirmed Time": zone.confirmed_time,

            "Pivot Time": zone.pivot_time,

            "Rejections": zone.rejections,

            "Strength": capped_strength,

            "Midline": (
                zone.low + zone.high
            ) / 2.0,

            "Strong": (
                zone.rejections >= strength
            ),

        })

    columns = [
        "Side",
        "Zone Low",
        "Zone High",
        "Confirmed Time",
        "Pivot Time",
        "Rejections",
        "Strength",
        "Midline",
        "Strong"
    ]

    if not result:
        return pd.DataFrame(columns=columns)

    output = pd.DataFrame(result)

    return output.sort_values(
        "Confirmed Time",
        ascending=False
    ).reset_index(drop=True)


# =========================================================
# END OF DESTINY AI S/R ENGINE
# =========================================================
