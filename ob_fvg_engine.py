from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PairedZone:
    direction: int
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

    @property
    def direction_label(self) -> str:
        if self.direction > 0:
            return "LONG"
        if self.direction < 0:
            return "SHORT"
        return "NEUTRAL"


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
    direction: Any,
) -> int:
    if isinstance(direction, str):
        value = direction.upper().strip()

        if value in {
            "LONG",
            "BUY",
            "BULL",
            "BULLISH",
        }:
            return 1

        if value in {
            "SHORT",
            "SELL",
            "BEAR",
            "BEARISH",
        }:
            return -1

        return 0

    try:
        numeric = float(direction)

        if numeric > 0:
            return 1

        if numeric < 0:
            return -1

    except (TypeError, ValueError):
        pass

    return 0


def _column(
    df: pd.DataFrame,
    *names: str,
) -> pd.Series | None:
    for name in names:
        if name in df.columns:
            return df[name]

    return None


def _ohlc(
    df: pd.DataFrame,
):
    open_col = _column(df, "Open", "open")
    high_col = _column(df, "High", "high")
    low_col = _column(df, "Low", "low")
    close_col = _column(df, "Close", "close")

    return (
        open_col,
        high_col,
        low_col,
        close_col,
    )


def _atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """
    Return a Wilder-style ATR series.

    mtf_runner.py expects this function to return
    a pandas Series rather than a scalar.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=float)

    _, high, low, close = _ohlc(df)

    if (
        high is None
        or low is None
        or close is None
    ):
        return pd.Series(
            index=df.index,
            dtype=float,
        )

    high = pd.to_numeric(
        high,
        errors="coerce",
    )

    low = pd.to_numeric(
        low,
        errors="coerce",
    )

    close = pd.to_numeric(
        close,
        errors="coerce",
    )

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    period = max(
        1,
        int(period),
    )

    return true_range.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=1,
    ).mean()


def _zone_from_values(
    direction: int,
    ob_low: float,
    ob_high: float,
    fvg_low: float,
    fvg_high: float,
    created_at: Any = None,
    strength: float = 0.0,
) -> PairedZone | None:
    ob_low = _safe_float(ob_low)
    ob_high = _safe_float(ob_high)
    fvg_low = _safe_float(fvg_low)
    fvg_high = _safe_float(fvg_high)

    if direction == 0:
        return None

    if (
        ob_high <= ob_low
        or fvg_high <= fvg_low
    ):
        return None

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
        strength=_safe_float(strength),
    )


def _zone_touched(
    zone: PairedZone,
    high: float,
    low: float,
) -> bool:
    return (
        high >= zone.lower
        and low <= zone.upper
    )


def _zone_mitigated(
    zone: PairedZone,
    close: float,
) -> bool:
    if zone.direction > 0:
        return close < zone.lower

    if zone.direction < 0:
        return close > zone.upper

    return False


def _find_order_blocks(
    df: pd.DataFrame,
    lookback: int = 100,
) -> list[dict[str, Any]]:
    if not isinstance(df, pd.DataFrame):
        return []

    if len(df) < 3:
        return []

    work = df.tail(
        max(3, int(lookback))
    ).copy()

    open_col, high_col, low_col, close_col = _ohlc(work)

    if any(
        value is None
        for value in (
            open_col,
            high_col,
            low_col,
            close_col,
        )
    ):
        return []

    opens = pd.to_numeric(
        open_col,
        errors="coerce",
    )

    highs = pd.to_numeric(
        high_col,
        errors="coerce",
    )

    lows = pd.to_numeric(
        low_col,
        errors="coerce",
    )

    closes = pd.to_numeric(
        close_col,
        errors="coerce",
    )

    atr = _atr(
        work,
        14,
    )

    blocks: list[dict[str, Any]] = []

    for i in range(1, len(work) - 1):
        current_open = _safe_float(
            opens.iloc[i]
        )
        current_high = _safe_float(
            highs.iloc[i]
        )
        current_low = _safe_float(
            lows.iloc[i]
        )
        current_close = _safe_float(
            closes.iloc[i]
        )

        next_close = _safe_float(
            closes.iloc[i + 1]
        )

        atr_value = _safe_float(
            atr.iloc[i],
            0.0,
        )

        if atr_value <= 0:
            continue

        # FIX: removed unused `body` variable (was dead code).

        displacement = abs(
            next_close - current_close
        )

        # Bullish order block:
        # bearish candle followed by bullish displacement.
        if (
            current_close < current_open
            and next_close > current_high
            and displacement >= atr_value * 0.25
        ):
            blocks.append(
                {
                    "direction": 1,
                    "low": current_low,
                    "high": current_high,
                    "created_at": work.index[i],
                    "created_bar": i,
                    "strength": (
                        displacement / atr_value
                    ),
                }
            )

        # Bearish order block:
        # bullish candle followed by bearish displacement.
        elif (
            current_close > current_open
            and next_close < current_low
            and displacement >= atr_value * 0.25
        ):
            blocks.append(
                {
                    "direction": -1,
                    "low": current_low,
                    "high": current_high,
                    "created_at": work.index[i],
                    "created_bar": i,
                    "strength": (
                        displacement / atr_value
                    ),
                }
            )

    return blocks


def _find_fvgs(
    df: pd.DataFrame,
    lookback: int = 100,
) -> list[dict[str, Any]]:
    if not isinstance(df, pd.DataFrame):
        return []

    if len(df) < 3:
        return []

    work = df.tail(
        max(3, int(lookback))
    ).copy()

    _, highs, lows, _ = _ohlc(work)

    if highs is None or lows is None:
        return []

    highs = pd.to_numeric(
        highs,
        errors="coerce",
    )

    lows = pd.to_numeric(
        lows,
        errors="coerce",
    )

    fvgs: list[dict[str, Any]] = []

    for i in range(2, len(work)):
        previous_high = _safe_float(
            highs.iloc[i - 2]
        )

        previous_low = _safe_float(
            lows.iloc[i - 2]
        )

        current_high = _safe_float(
            highs.iloc[i]
        )

        current_low = _safe_float(
            lows.iloc[i]
        )

        # Bullish FVG.
        if current_low > previous_high:
            fvgs.append(
                {
                    "direction": 1,
                    "low": previous_high,
                    "high": current_low,
                    "created_at": work.index[i],
                    "created_bar": i,
                }
            )

        # Bearish FVG.
        elif current_high < previous_low:
            fvgs.append(
                {
                    "direction": -1,
                    "low": current_high,
                    "high": previous_low,
                    "created_at": work.index[i],
                    "created_bar": i,
                }
            )

    return fvgs


def _pair_order_blocks_and_fvgs(
    order_blocks: list[dict[str, Any]],
    fvgs: list[dict[str, Any]],
    max_zones: int = 20,
    max_pair_distance: int = 50,
) -> list[PairedZone]:
    """
    Pair OBs with FVGs of the same direction.

    FIX: a pairing now requires the FVG to form at or after the OB
    (within `max_pair_distance` bars). Previously a stale OB could
    pair with a fresh FVG created many bars later, producing
    nonsensical zones.
    """
    paired: list[PairedZone] = []

    for ob in reversed(order_blocks):
        direction = _normalise_direction(
            ob.get("direction")
        )

        if direction == 0:
            continue

        ob_low = _safe_float(
            ob.get("low")
        )

        ob_high = _safe_float(
            ob.get("high")
        )

        ob_bar = ob.get("created_bar")

        for fvg in reversed(fvgs):
            fvg_direction = _normalise_direction(
                fvg.get("direction")
            )

            if fvg_direction != direction:
                continue

            # FIX: enforce temporal ordering.
            fvg_bar = fvg.get("created_bar")

            if (
                ob_bar is not None
                and fvg_bar is not None
            ):
                if fvg_bar < ob_bar:
                    continue

                if (
                    fvg_bar - ob_bar
                    > max_pair_distance
                ):
                    continue

            fvg_low = _safe_float(
                fvg.get("low")
            )

            fvg_high = _safe_float(
                fvg.get("high")
            )

            overlap_low = max(
                ob_low,
                fvg_low,
            )

            overlap_high = min(
                ob_high,
                fvg_high,
            )

            if overlap_high <= overlap_low:
                continue

            zone = _zone_from_values(
                direction=direction,
                ob_low=ob_low,
                ob_high=ob_high,
                fvg_low=fvg_low,
                fvg_high=fvg_high,
                created_at=fvg.get(
                    "created_at",
                    ob.get("created_at"),
                ),
                strength=max(
                    _safe_float(
                        ob.get("strength")
                    ),
                    0.0,
                ),
            )

            if zone is None:
                continue

            paired.append(zone)
            break

        if len(paired) >= max_zones:
            break

    return paired


def _structure_state(
    df: pd.DataFrame,
    pivot_length: int = 2,
) -> dict[str, Any]:
    """
    FIX: previously this used `highs.iloc[-3]` as the "recent high"
    (an arbitrary offset). It now uses the latest CONFIRMED swing
    high/low (a pivot confirmed by `pivot_length` bars on each side).
    BOS is "last close > last confirmed swing high" (or below the
    swing low).
    """
    empty = {
        "trend": "NEUTRAL",
        "trend_direction": 0,
        "bull_bos": False,
        "bear_bos": False,
        "new_bull_setup": False,
        "new_bear_setup": False,
    }

    if not isinstance(df, pd.DataFrame):
        return empty

    # Need enough bars for a pivot on each side plus a close.
    if len(df) < pivot_length * 2 + 2:
        return empty

    _, highs, lows, closes = _ohlc(df)

    if (
        highs is None
        or lows is None
        or closes is None
    ):
        return empty

    highs = pd.to_numeric(
        highs,
        errors="coerce",
    )

    lows = pd.to_numeric(
        lows,
        errors="coerce",
    )

    closes = pd.to_numeric(
        closes,
        errors="coerce",
    )

    highs_arr = highs.to_numpy(dtype=float)
    lows_arr = lows.to_numpy(dtype=float)
    closes_arr = closes.to_numpy(dtype=float)

    n = len(highs_arr)

    # FIX: find the latest confirmed swing high/low (a pivot with
    # `pivot_length` bars on each side). Previously the code used
    # highs.iloc[-3] which is not a pivot at all.
    last_swing_high: float | None = None
    last_swing_low: float | None = None

    for p in range(n - pivot_length - 1, pivot_length - 1, -1):
        window_h = highs_arr[p - pivot_length:p + pivot_length + 1]
        window_l = lows_arr[p - pivot_length:p + pivot_length + 1]

        if (
            np.isfinite(window_h).all()
            and highs_arr[p] == window_h.max()
            and (window_h == highs_arr[p]).sum() == 1
        ):
            last_swing_high = float(highs_arr[p])
            break

    for p in range(n - pivot_length - 1, pivot_length - 1, -1):
        window_l = lows_arr[p - pivot_length:p + pivot_length + 1]

        if (
            np.isfinite(window_l).all()
            and lows_arr[p] == window_l.min()
            and (window_l == lows_arr[p]).sum() == 1
        ):
            last_swing_low = float(lows_arr[p])
            break

    last_close = closes_arr[-1] if n else float("nan")

    bull_bos = (
        last_swing_high is not None
        and np.isfinite(last_close)
        and last_close > last_swing_high
    )

    bear_bos = (
        last_swing_low is not None
        and np.isfinite(last_close)
        and last_close < last_swing_low
    )

    if bull_bos and not bear_bos:
        direction = 1
        trend = "BULLISH"
    elif bear_bos and not bull_bos:
        direction = -1
        trend = "BEARISH"
    else:
        # Fall back to last-bar direction.
        if n >= 2 and np.isfinite(closes_arr[-1]) and np.isfinite(closes_arr[-2]):
            if closes_arr[-1] > closes_arr[-2]:
                direction = 1
                trend = "BULLISH"
            elif closes_arr[-1] < closes_arr[-2]:
                direction = -1
                trend = "BEARISH"
            else:
                direction = 0
                trend = "NEUTRAL"
        else:
            direction = 0
            trend = "NEUTRAL"

    return {
        "trend": trend,
        "trend_direction": direction,
        "bull_bos": bool(bull_bos),
        "bear_bos": bool(bear_bos),
        "new_bull_setup": bool(bull_bos),
        "new_bear_setup": bool(bear_bos),
    }


def _compute_trend_series(
    df: pd.DataFrame,
    pivot_length: int = 2,
) -> pd.Series:
    """
    FIX: build a PER-BAR trend direction series.

    Previously the code broadcast the scalar `structure["trend_direction"]`
    across every row of `signals_df`. mtf_runner._get_trend reads
    `signals_df["trend_direction"].iloc[-1]`, so it was silently
    getting a constant. Now the column is a genuine rolling series.

    Direction at bar i is:
      +1 if close[i] > latest confirmed swing high before i
      -1 if close[i] < latest confirmed swing low before i
       0 otherwise (no confirmed pivot or price is inside the range)
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=int)

    _, highs, lows, closes = _ohlc(df)

    if (
        highs is None
        or lows is None
        or closes is None
    ):
        return pd.Series(
            index=df.index,
            dtype=int,
        )

    highs = pd.to_numeric(highs, errors="coerce")
    lows = pd.to_numeric(lows, errors="coerce")
    closes = pd.to_numeric(closes, errors="coerce")

    highs_arr = highs.to_numpy(dtype=float)
    lows_arr = lows.to_numpy(dtype=float)
    closes_arr = closes.to_numpy(dtype=float)

    n = len(highs_arr)
    direction = np.zeros(n, dtype=int)

    # For each bar i, find the most recent confirmed swing high/low
    # that existed BEFORE i (i.e. confirmed by pivot_length bars on
    # each side and the right side is fully before i).
    last_swing_high: float | None = None
    last_swing_low: float | None = None

    for i in range(n):
        # At bar i, the "newly confirmed" pivot is p = i - pivot_length - 1
        # (needs pivot_length bars to the right = i - 1).
        p = i - pivot_length - 1

        if p >= pivot_length:
            window_h = highs_arr[p - pivot_length:p + pivot_length + 1]
            window_l = lows_arr[p - pivot_length:p + pivot_length + 1]

            if (
                np.isfinite(window_h).all()
                and highs_arr[p] == window_h.max()
                and (window_h == highs_arr[p]).sum() == 1
            ):
                last_swing_high = float(highs_arr[p])

            if (
                np.isfinite(window_l).all()
                and lows_arr[p] == window_l.min()
                and (window_l == lows_arr[p]).sum() == 1
            ):
                last_swing_low = float(lows_arr[p])

        if not np.isfinite(closes_arr[i]):
            continue

        if last_swing_high is not None and closes_arr[i] > last_swing_high:
            direction[i] = 1
        elif last_swing_low is not None and closes_arr[i] < last_swing_low:
            direction[i] = -1

    return pd.Series(direction, index=df.index, dtype=int)


def _build_bos_series(
    df: pd.DataFrame,
    direction_series: pd.Series,
) -> dict[str, pd.Series]:
    """
    FIX: build per-bar BOS / setup series.

    Previously the code only set the last row of these columns, so
    mtf_runner._safe_last_bool("bull_bos", lookback=5) was effectively
    reading 4 False bars + 1 possibly-True bar. Now these are real
    per-bar booleans.
    """
    index = df.index

    # A BOS event fires on the bar where direction_series first
    # transitions into a bull/bear state (i.e. the bar where close
    # crossed the most recent confirmed swing high/low).
    bull_bos = (
        (direction_series == 1)
        & (direction_series.shift(1) != 1)
    )

    bear_bos = (
        (direction_series == -1)
        & (direction_series.shift(1) != -1)
    )

    bull_bos = bull_bos.reindex(index, fill_value=False).astype(bool)
    bear_bos = bear_bos.reindex(index, fill_value=False).astype(bool)

    return {
        "bull_bos": bull_bos,
        "bear_bos": bear_bos,
        "new_bull_setup": bull_bos,
        "new_bear_setup": bear_bos,
    }


def _detect_liquidity_sweep(
    df: pd.DataFrame,
    lookback: int = 5,
) -> pd.Series:
    if not isinstance(df, pd.DataFrame):
        return pd.Series(
            dtype=int
        )

    if df.empty:
        return pd.Series(
            index=df.index,
            dtype=int,
        )

    _, highs, lows, closes = _ohlc(df)

    if (
        highs is None
        or lows is None
        or closes is None
    ):
        return pd.Series(
            index=df.index,
            dtype=int,
        )

    highs = pd.to_numeric(
        highs,
        errors="coerce",
    )

    lows = pd.to_numeric(
        lows,
        errors="coerce",
    )

    closes = pd.to_numeric(
        closes,
        errors="coerce",
    )

    lookback = max(
        2,
        int(lookback),
    )

    sweeps = pd.Series(
        0,
        index=df.index,
        dtype=int,
    )

    for i in range(lookback, len(df)):
        prior_high = highs.iloc[
            i - lookback:i
        ].max()

        prior_low = lows.iloc[
            i - lookback:i
        ].min()

        current_high = highs.iloc[i]
        current_low = lows.iloc[i]
        current_close = closes.iloc[i]

        if (
            pd.notna(prior_high)
            and pd.notna(current_high)
            and pd.notna(current_close)
            and current_high > prior_high
            and current_close < prior_high
        ):
            sweeps.iloc[i] += 1

        if (
            pd.notna(prior_low)
            and pd.notna(current_low)
            and pd.notna(current_close)
            and current_low < prior_low
            and current_close > prior_low
        ):
            sweeps.iloc[i] += 1

    return sweeps


def run_ob_fvg_engine(
    df: pd.DataFrame,
    lookback: int = 100,
    max_zones: int = 20,
    timeframe_minutes: int | None = None,
) -> dict[str, Any]:
    """
    OB/FVG engine used by mtf_runner.py.

    timeframe_minutes is accepted for compatibility with
    the 5-timeframe cascade.
    """

    if not isinstance(df, pd.DataFrame):
        return {
            "paired_zones": [],
            "active_paired_zones": [],
            "active_zones": [],
            "order_blocks": [],
            "fvgs": [],
            "atr": pd.Series(dtype=float),
            "trend": "NEUTRAL",
            "trend_direction": 0,
            "bull_bos": False,
            "bear_bos": False,
            "new_bull_setup": False,
            "new_bear_setup": False,
            "signals_df": pd.DataFrame(),
        }

    if df.empty:
        return {
            "paired_zones": [],
            "active_paired_zones": [],
            "active_zones": [],
            "order_blocks": [],
            "fvgs": [],
            "atr": pd.Series(
                index=df.index,
                dtype=float,
            ),
            "trend": "NEUTRAL",
            "trend_direction": 0,
            "bull_bos": False,
            "bear_bos": False,
            "new_bull_setup": False,
            "new_bear_setup": False,
            "signals_df": pd.DataFrame(
                index=df.index
            ),
        }

    work = df.tail(
        max(3, int(lookback))
    ).copy()

    atr_series = _atr(
        work,
        14,
    )

    order_blocks = _find_order_blocks(
        work,
        lookback=lookback,
    )

    fvgs = _find_fvgs(
        work,
        lookback=lookback,
    )

    paired_zones = _pair_order_blocks_and_fvgs(
        order_blocks=order_blocks,
        fvgs=fvgs,
        max_zones=max_zones,
    )

    _, highs, lows, closes = _ohlc(work)

    if (
        highs is None
        or lows is None
        or closes is None
    ):
        highs = pd.Series(
            index=work.index,
            dtype=float,
        )
        lows = pd.Series(
            index=work.index,
            dtype=float,
        )
        closes = pd.Series(
            index=work.index,
            dtype=float,
        )

    highs = pd.to_numeric(
        highs,
        errors="coerce",
    )

    lows = pd.to_numeric(
        lows,
        errors="coerce",
    )

    closes = pd.to_numeric(
        closes,
        errors="coerce",
    )

    active_zones: list[PairedZone] = []

    for zone in paired_zones:
        zone.active = True
        zone.mitigated = False

        if len(work) > 0:
            last_high = _safe_float(
                highs.iloc[-1]
            )

            last_low = _safe_float(
                lows.iloc[-1]
            )

            last_close = _safe_float(
                closes.iloc[-1]
            )

            if _zone_touched(
                zone,
                last_high,
                last_low,
            ):
                if _zone_mitigated(
                    zone,
                    last_close,
                ):
                    zone.mitigated = True
                    zone.active = False

        if zone.active:
            active_zones.append(zone)

    structure = _structure_state(
        work
    )

    # FIX: per-bar trend direction (was a constant broadcast).
    trend_series = _compute_trend_series(work)

    # FIX: per-bar BOS / setup flags (was only set on the last row).
    bos_series = _build_bos_series(work, trend_series)

    sweep_series = _detect_liquidity_sweep(
        work,
        lookback=5,
    )

    # FIX: build signals_df cleanly from Series. The old approach used
    # .iloc[-1, get_loc(...)] assignments which is verbose and error
    # prone. Now every column is a real per-bar series.
    signals_df = pd.DataFrame(
        index=work.index,
    )

    signals_df["trend_direction"] = trend_series.reindex(
        work.index
    ).fillna(0).astype(int)

    signals_df["trend"] = signals_df["trend_direction"].map(
        {1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}
    )

    signals_df["bull_bos"] = bos_series["bull_bos"].reindex(
        work.index
    ).fillna(False).astype(bool)

    signals_df["bear_bos"] = bos_series["bear_bos"].reindex(
        work.index
    ).fillna(False).astype(bool)

    signals_df["new_bull_setup"] = bos_series["new_bull_setup"].reindex(
        work.index
    ).fillna(False).astype(bool)

    signals_df["new_bear_setup"] = bos_series["new_bear_setup"].reindex(
        work.index
    ).fillna(False).astype(bool)

    signals_df["swept_liquidity_count"] = sweep_series.reindex(
        work.index
    ).fillna(0).astype(int)

    return {
        "paired_zones": paired_zones,
        "active_paired_zones": active_zones,

        # mtf_runner.py expects this exact key.
        "active_zones": active_zones,

        "order_blocks": order_blocks,
        "fvgs": fvgs,
        "atr": atr_series,

        "trend": structure["trend"],
        "trend_direction": structure["trend_direction"],

        "bull_bos": structure["bull_bos"],
        "bear_bos": structure["bear_bos"],
        "new_bull_setup": structure["new_bull_setup"],
        "new_bear_setup": structure["new_bear_setup"],

        # Required by mtf_runner.py.
        "signals_df": signals_df,

        "timeframe_minutes": (
            timeframe_minutes
        ),
}
