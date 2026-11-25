from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass
class PairedZone:
    """
    Represents an Order Block / Fair Value Gap paired zone.
    """

    direction: str
    low: float
    high: float
    ob_low: float
    ob_high: float
    fvg_low: float
    fvg_high: float
    created_at: Any = None
    active: bool = True
    mitigated: bool = False
    strength: float = 0.0

    @property
    def lower(self) -> float:
        return min(self.low, self.high)

    @property
    def upper(self) -> float:
        return max(self.low, self.high)

    @property
    def midpoint(self) -> float:
        return (self.lower + self.upper) / 2.0


# ============================================================
# SAFE HELPERS
# ============================================================

def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def _normalise_direction(
    value: Any,
) -> str:
    if isinstance(value, str):
        text = value.upper().strip()

        if text in {
            "LONG",
            "BUY",
            "BULL",
            "BULLISH",
        }:
            return "LONG"

        if text in {
            "SHORT",
            "SELL",
            "BEAR",
            "BEARISH",
        }:
            return "SHORT"

    try:
        numeric = float(value)

        if numeric > 0:
            return "LONG"

        if numeric < 0:
            return "SHORT"

    except (TypeError, ValueError):
        pass

    return ""


def _column(
    df: pd.DataFrame,
    *names: str,
) -> str | None:
    lookup = {
        str(column).lower(): column
        for column in df.columns
    }

    for name in names:
        found = lookup.get(name.lower())

        if found is not None:
            return found

    return None


def _ohlc(
    df: pd.DataFrame,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
] | None:
    open_col = _column(df, "open")
    high_col = _column(df, "high")
    low_col = _column(df, "low")
    close_col = _column(df, "close")

    if not all(
        column is not None
        for column in (
            open_col,
            high_col,
            low_col,
            close_col,
        )
    ):
        return None

    return (
        pd.to_numeric(
            df[open_col],
            errors="coerce",
        ),
        pd.to_numeric(
            df[high_col],
            errors="coerce",
        ),
        pd.to_numeric(
            df[low_col],
            errors="coerce",
        ),
        pd.to_numeric(
            df[close_col],
            errors="coerce",
        ),
    )


# ============================================================
# ATR
# ============================================================

def _atr(
    df: pd.DataFrame,
    period: int = 14,
) -> float:
    """
    Calculate the latest Average True Range.

    Returns 0.0 when there is insufficient valid data.
    """
    if not isinstance(df, pd.DataFrame):
        return 0.0

    if df.empty:
        return 0.0

    ohlc = _ohlc(df)

    if ohlc is None:
        return 0.0

    _, high, low, close = ohlc

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_series = true_range.rolling(
        window=max(1, int(period)),
        min_periods=1,
    ).mean()

    if atr_series.empty:
        return 0.0

    value = atr_series.iloc[-1]

    return _safe_float(value)


# ============================================================
# ZONE HELPERS
# ============================================================

def _zone_from_values(
    direction: str,
    ob_low: float,
    ob_high: float,
    fvg_low: float,
    fvg_high: float,
    created_at: Any = None,
    strength: float = 0.0,
) -> PairedZone | None:
    if direction not in {
        "LONG",
        "SHORT",
    }:
        return None

    ob_low = _safe_float(ob_low)
    ob_high = _safe_float(ob_high)
    fvg_low = _safe_float(fvg_low)
    fvg_high = _safe_float(fvg_high)

    if ob_low <= 0 or ob_high <= 0:
        return None

    if fvg_low <= 0 or fvg_high <= 0:
        return None

    ob_low, ob_high = sorted(
        (ob_low, ob_high)
    )

    fvg_low, fvg_high = sorted(
        (fvg_low, fvg_high)
    )

    low = max(
        ob_low,
        fvg_low,
    )

    high = min(
        ob_high,
        fvg_high,
    )

    if high <= low:
        return None

    return PairedZone(
        direction=direction,
        low=low,
        high=high,
        ob_low=ob_low,
        ob_high=ob_high,
        fvg_low=fvg_low,
        fvg_high=fvg_high,
        created_at=created_at,
        active=True,
        mitigated=False,
        strength=_safe_float(
            strength
        ),
    )


def _zone_touched(
    zone: PairedZone,
    candle_low: float,
    candle_high: float,
) -> bool:
    return not (
        candle_high < zone.lower
        or candle_low > zone.upper
    )


def _zone_mitigated(
    zone: PairedZone,
    close: float,
) -> bool:
    """
    A conservative mitigation rule.

    LONG zone:
        closes below the zone.

    SHORT zone:
        closes above the zone.
    """
    if zone.direction == "LONG":
        return close < zone.lower

    if zone.direction == "SHORT":
        return close > zone.upper

    return False


# ============================================================
# ORDER BLOCK DETECTION
# ============================================================

def _find_order_blocks(
    df: pd.DataFrame,
    lookback: int = 100,
) -> list[dict[str, Any]]:
    """
    Detect simple displacement-based Order Blocks.

    Bullish OB:
        bearish candle followed by bullish displacement.

    Bearish OB:
        bullish candle followed by bearish displacement.
    """
    ohlc = _ohlc(df)

    if ohlc is None:
        return []

    open_, high, low, close = ohlc

    atr_value = _atr(df)

    if atr_value <= 0:
        return []

    start = max(
        1,
        len(df) - max(10, int(lookback)),
    )

    zones: list[dict[str, Any]] = []

    for i in range(
        start,
        len(df) - 1,
    ):
        current_open = _safe_float(
            open_.iloc[i]
        )
        current_high = _safe_float(
            high.iloc[i]
        )
        current_low = _safe_float(
            low.iloc[i]
        )
        current_close = _safe_float(
            close.iloc[i]
        )

        next_open = _safe_float(
            open_.iloc[i + 1]
        )
        next_close = _safe_float(
            close.iloc[i + 1]
        )
        next_high = _safe_float(
            high.iloc[i + 1]
        )
        next_low = _safe_float(
            low.iloc[i + 1]
        )

        current_body = abs(
            current_close - current_open
        )

        next_body = abs(
            next_close - next_open
        )

        if current_body <= 0:
            continue

        # Bullish displacement from a bearish candle.
        if (
            current_close < current_open
            and next_close > next_open
            and next_body >= atr_value * 0.35
            and next_close > current_high
        ):
            zones.append(
                {
                    "direction": "LONG",
                    "low": current_low,
                    "high": current_high,
                    "index": i,
                    "created_at": df.index[i],
                    "strength": next_body / atr_value,
                }
            )

        # Bearish displacement from a bullish candle.
        elif (
            current_close > current_open
            and next_close < next_open
            and next_body >= atr_value * 0.35
            and next_close < current_low
        ):
            zones.append(
                {
                    "direction": "SHORT",
                    "low": current_low,
                    "high": current_high,
                    "index": i,
                    "created_at": df.index[i],
                    "strength": next_body / atr_value,
                }
            )

    return zones


# ============================================================
# FAIR VALUE GAP DETECTION
# ============================================================

def _find_fvgs(
    df: pd.DataFrame,
    lookback: int = 100,
) -> list[dict[str, Any]]:
    """
    Detect three-candle Fair Value Gaps.

    Bullish FVG:
        candle 3 low > candle 1 high.

    Bearish FVG:
        candle 3 high < candle 1 low.
    """
    ohlc = _ohlc(df)

    if ohlc is None:
        return []

    _, high, low, close = ohlc

    atr_value = _atr(df)

    if atr_value <= 0:
        return []

    start = max(
        1,
        len(df) - max(10, int(lookback)),
    )

    zones: list[dict[str, Any]] = []

    for i in range(
        start,
        len(df) - 1,
    ):
        if i + 1 >= len(df):
            break

        first_high = _safe_float(
            high.iloc[i - 1]
        )

        first_low = _safe_float(
            low.iloc[i - 1]
        )

        third_high = _safe_float(
            high.iloc[i + 1]
        )

        third_low = _safe_float(
            low.iloc[i + 1]
        )

        # Bullish FVG.
        if third_low > first_high:
            gap_low = first_high
            gap_high = third_low

            if (
                gap_high - gap_low
                >= atr_value * 0.05
            ):
                zones.append(
                    {
                        "direction": "LONG",
                        "low": gap_low,
                        "high": gap_high,
                        "index": i,
                        "created_at": df.index[i],
                    }
                )

        # Bearish FVG.
        elif third_high < first_low:
            gap_low = third_high
            gap_high = first_low

            if (
                gap_high - gap_low
                >= atr_value * 0.05
            ):
                zones.append(
                    {
                        "direction": "SHORT",
                        "low": gap_low,
                        "high": gap_high,
                        "index": i,
                        "created_at": df.index[i],
                    }
                )

    return zones


# ============================================================
# PAIR OB + FVG
# ============================================================

def _pair_order_blocks_and_fvgs(
    order_blocks: list[dict[str, Any]],
    fvgs: list[dict[str, Any]],
) -> list[PairedZone]:
    """
    Pair an Order Block with an FVG when they overlap.

    The pair must have the same directional bias.
    """
    paired: list[PairedZone] = []

    for ob in order_blocks:
        direction = _normalise_direction(
            ob.get("direction")
        )

        if not direction:
            continue

        ob_low = _safe_float(
            ob.get("low")
        )

        ob_high = _safe_float(
            ob.get("high")
        )

        ob_index = int(
            ob.get("index", 0)
        )

        for fvg in fvgs:
            fvg_direction = _normalise_direction(
                fvg.get("direction")
            )

            if fvg_direction != direction:
                continue

            fvg_index = int(
                fvg.get("index", 0)
            )

            # Require the FVG to form at or after
            # the Order Block candle.
            if fvg_index < ob_index:
                continue

            zone = _zone_from_values(
                direction=direction,
                ob_low=ob_low,
                ob_high=ob_high,
                fvg_low=_safe_float(
                    fvg.get("low")
                ),
                fvg_high=_safe_float(
                    fvg.get("high")
                ),
                created_at=fvg.get(
                    "created_at",
                    ob.get("created_at"),
                ),
                strength=_safe_float(
                    ob.get("strength", 0.0)
                ),
            )

            if zone is not None:
                paired.append(zone)

    return paired


# ============================================================
# STRUCTURE
# ============================================================

def _structure_state(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Lightweight BOS / directional structure detection.
    """
    ohlc = _ohlc(df)

    if ohlc is None or len(df) < 6:
        return {
            "trend": "NEUTRAL",
            "bull_bos": False,
            "bear_bos": False,
            "new_bull_setup": False,
            "new_bear_setup": False,
        }

    _, high, low, close = ohlc

    recent_high = _safe_float(
        high.iloc[-2]
    )

    recent_low = _safe_float(
        low.iloc[-2]
    )

    prior_high = _safe_float(
        high.iloc[-5:-2].max()
    )

    prior_low = _safe_float(
        low.iloc[-5:-2].min()
    )

    latest_close = _safe_float(
        close.iloc[-1]
    )

    bull_bos = (
        latest_close > prior_high
        and latest_close > recent_high
    )

    bear_bos = (
        latest_close < prior_low
        and latest_close < recent_low
    )

    if bull_bos:
        trend = "BULLISH"
    elif bear_bos:
        trend = "BEARISH"
    elif latest_close > prior_high:
        trend = "BULLISH"
    elif latest_close < prior_low:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    return {
        "trend": trend,
        "bull_bos": bool(bull_bos),
        "bear_bos": bool(bear_bos),
        "new_bull_setup": bool(bull_bos),
        "new_bear_setup": bool(bear_bos),
    }


# ============================================================
# MAIN ENGINE
# ============================================================

def run_ob_fvg_engine(
    df: pd.DataFrame,
    lookback: int = 100,
    max_zones: int = 20,
) -> dict[str, Any]:
    """
    Run the Order Block + Fair Value Gap engine.

    Returns a dictionary consumed by mtf_runner.
    """
    if not isinstance(
        df,
        pd.DataFrame,
    ):
        return {
            "paired_zones": [],
            "active_paired_zones": [],
            "order_blocks": [],
            "fvgs": [],
            "atr": 0.0,
            "trend": "NEUTRAL",
            "bull_bos": False,
            "bear_bos": False,
            "new_bull_setup": False,
            "new_bear_setup": False,
        }

    if df.empty:
        return {
            "paired_zones": [],
            "active_paired_zones": [],
            "order_blocks": [],
            "fvgs": [],
            "atr": 0.0,
            "trend": "NEUTRAL",
            "bull_bos": False,
            "bear_bos": False,
            "new_bull_setup": False,
            "new_bear_setup": False,
        }

    ohlc = _ohlc(df)

    if ohlc is None:
        return {
            "paired_zones": [],
            "active_paired_zones": [],
            "order_blocks": [],
            "fvgs": [],
            "atr": 0.0,
            "trend": "NEUTRAL",
            "bull_bos": False,
            "bear_bos": False,
            "new_bull_setup": False,
            "new_bear_setup": False,
        }

    _, high, low, close = ohlc

    atr_value = _atr(df)

    order_blocks = _find_order_blocks(
        df,
        lookback=lookback,
    )

    fvgs = _find_fvgs(
        df,
        lookback=lookback,
    )

    paired_zones = _pair_order_blocks_and_fvgs(
        order_blocks,
        fvgs,
    )

    # Keep the most recent zones.
    paired_zones = paired_zones[
        -max(1, int(max_zones)):
    ]

    active_zones: list[PairedZone] = []

    latest_close = _safe_float(
        close.iloc[-1]
    )

    latest_high = _safe_float(
        high.iloc[-1]
    )

    latest_low = _safe_float(
        low.iloc[-1]
    )

    for zone in paired_zones:
        zone.mitigated = _zone_mitigated(
            zone,
            latest_close,
        )

        if zone.mitigated:
            zone.active = False

        if _zone_touched(
            zone,
            latest_low,
            latest_high,
        ):
            # Interaction does not automatically invalidate
            # the zone. A closing break does.
            pass

        if zone.active:
            active_zones.append(zone)

    structure = _structure_state(df)

    return {
        "paired_zones": paired_zones,
        "active_paired_zones": active_zones,
        "order_blocks": order_blocks,
        "fvgs": fvgs,
        "atr": atr_value,
        "trend": structure["trend"],
        "bull_bos": structure["bull_bos"],
        "bear_bos": structure["bear_bos"],
        "new_bull_setup": structure[
            "new_bull_setup"
        ],
        "new_bear_setup": structure[
            "new_bear_setup"
        ],
  }
