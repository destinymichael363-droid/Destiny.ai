from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from mtf_runner import run_closed_candle_5tf_cascade


@dataclass
class TradeSetup:
    symbol: str
    tf_label: str
    direction: str
    entry: float
    sl: float
    tp: float
    rr: float
    confidence: str
    reason: str


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Safely convert a value to float."""

    try:
        if value is None:
            return default

        if isinstance(value, bool):
            return default

        result = float(value)

        if pd.isna(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def _normalise_direction(
    direction: Any,
) -> str:
    """Convert common directional formats into LONG/SHORT."""

    if isinstance(direction, str):
        value = direction.upper().strip()

        if value in {
            "LONG",
            "BUY",
            "BULL",
            "BULLISH",
        }:
            return "LONG"

        if value in {
            "SHORT",
            "SELL",
            "BEAR",
            "BEARISH",
        }:
            return "SHORT"

    try:
        numeric = int(direction)

        if numeric > 0:
            return "LONG"

        if numeric < 0:
            return "SHORT"

    except (TypeError, ValueError):
        pass

    return ""


def _validate_prices(
    direction: str,
    entry: float,
    sl: float,
    tp: float,
) -> bool:
    """Validate that entry, SL and TP form a logical trade."""

    if (
        entry <= 0
        or sl <= 0
        or tp <= 0
    ):
        return False

    if direction == "LONG":
        return (
            sl < entry
            and tp > entry
        )

    if direction == "SHORT":
        return (
            sl > entry
            and tp < entry
        )

    return False


def _calculate_rr(
    direction: str,
    entry: float,
    sl: float,
    tp: float,
) -> float:
    """Calculate actual reward-to-risk."""

    if direction == "LONG":
        risk = entry - sl
        reward = tp - entry

    elif direction == "SHORT":
        risk = sl - entry
        reward = entry - tp

    else:
        return 0.0

    if risk <= 0:
        return 0.0

    return reward / risk


def _build_reason(
    cascade: dict,
    signal: Any,
) -> str:
    """
    Build a transparent explanation for the trade.

    FIX: the "1H confirmation aligned" line was hardcoded and would
    appear even when the 1H trend was neutral. Now it reads the real
    1H trend from the cascade and reports 'aligned' or 'neutral'
    accurately. Format stays short (Option A).
    """

    primary = cascade.get(
        "primary_setup",
        {},
    )

    execution = cascade.get(
        "execution_confirmation",
        {},
    )

    recent_sweeps = cascade.get(
        "recent_sweeps",
        {},
    )

    tf_trends = cascade.get(
        "tf_trends",
        {},
    )

    htf_bias = cascade.get(
        "htf_bias_label",
        "NEUTRAL",
    )

    structure_confirmed = bool(
        primary.get(
            "structure_confirmation",
            False,
        )
    )

    execution_tf = execution.get(
        "tf"
    )

    m15_sweeps = int(
        recent_sweeps.get(
            "15M",
            0,
        )
    )

    m5_sweeps = int(
        recent_sweeps.get(
            "5M",
            0,
        )
    )

    m1_sweeps = int(
        recent_sweeps.get(
            "1M",
            0,
        )
    )

    # FIX: report actual 1H state rather than hardcoding "aligned".
    t_1h = int(tf_trends.get("1H", 0))

    if t_1h == 1:
        one_h = "BULLISH"
    elif t_1h == -1:
        one_h = "BEARISH"
    else:
        one_h = "NEUTRAL"

    components = [
        f"4H bias: {htf_bias}",
        f"1H: {one_h}",
        "15M HTF/OB-FVG overlap",
    ]

    if structure_confirmed:
        components.append(
            "15M structure confirmed"
        )

    if execution_tf:
        components.append(
            f"{execution_tf} execution confirmation"
        )

    if m15_sweeps > 0:
        components.append(
            f"15M liquidity sweep ({m15_sweeps})"
        )

    if m5_sweeps > 0:
        components.append(
            f"5M liquidity sweep ({m5_sweeps})"
        )

    if m1_sweeps > 0:
        components.append(
            f"1M liquidity sweep ({m1_sweeps})"
        )

    return " | ".join(components)


def _cascade_to_setup(
    cascade: dict,
    symbol: str,
    tf_label: str = "15M",
    rr_target: float = 5.0,
) -> TradeSetup | None:
    """
    Convert a validated MTF cascade signal into
    the TradeSetup object used by the application.
    """

    signal = cascade.get(
        "scalp_signal"
    )

    if signal is None:
        return None

    direction = _normalise_direction(
        getattr(
            signal,
            "direction",
            "",
        )
    )

    entry = _safe_float(
        getattr(
            signal,
            "entry_price",
            0.0,
        )
    )

    sl = _safe_float(
        getattr(
            signal,
            "stop_loss",
            0.0,
        )
    )

    tp = _safe_float(
        getattr(
            signal,
            "take_profit",
            0.0,
        )
    )

    if not _validate_prices(
        direction,
        entry,
        sl,
        tp,
    ):
        return None

    actual_rr = _calculate_rr(
        direction,
        entry,
        sl,
        tp,
    )

    if actual_rr <= 0:
        return None

    # FIX: previously the tolerance was rr_target * 0.99, which
    # meant a 4.95 RR requirement at rr_target=5.0. Because the
    # cascade passes ROUNDED entry/sl/tp into ScalpSignal, the RR
    # recomputed here can differ from the RR mtf_runner already
    # validated by a small rounding error. That caused valid
    # signals to be silently dropped.
    #
    # Now we allow 10% slack — enough to absorb rounding, tight
    # enough to still catch real math bugs.
    minimum_rr = max(
        1.0,
        rr_target * 0.90,
    )

    if actual_rr < minimum_rr:
        return None

    confidence = str(
        getattr(
            signal,
            "confidence",
            "STANDARD",
        )
    )

    reason = _build_reason(
        cascade,
        signal,
    )

    return TradeSetup(
        symbol=symbol,
        tf_label=tf_label,
        direction=direction,
        entry=round(entry, 5),
        sl=round(sl, 5),
        tp=round(tp, 5),
        rr=round(actual_rr, 3),
        confidence=confidence,
        reason=reason,
    )


def _prepare_mtf_frames(
    df: pd.DataFrame,
    df_1h: pd.DataFrame | None,
    df_4h: pd.DataFrame | None,
    mtf_data: dict[str, pd.DataFrame] | None,
) -> dict[str, pd.DataFrame]:
    """
    Assemble the timeframe dictionary expected by mtf_runner.
    """

    frames: dict[str, pd.DataFrame] = {}

    if mtf_data:
        for tf in (
            "4H",
            "1H",
            "15M",
            "5M",
            "1M",
        ):
            value = mtf_data.get(tf)

            if isinstance(
                value,
                pd.DataFrame,
            ):
                frames[tf] = value

    if isinstance(
        df_4h,
        pd.DataFrame,
    ):
        frames["4H"] = df_4h

    if isinstance(
        df_1h,
        pd.DataFrame,
    ):
        frames["1H"] = df_1h

    if isinstance(
        df,
        pd.DataFrame,
    ):
        frames["15M"] = df

    return frames


def _has_required_timeframes(
    frames: dict[str, pd.DataFrame],
) -> bool:
    """
    Confirm that all five strategy timeframes are available.
    """

    required = (
        "4H",
        "1H",
        "15M",
        "5M",
        "1M",
    )

    return all(
        isinstance(
            frames.get(tf),
            pd.DataFrame,
        )
        and not frames[tf].empty
        for tf in required
    )


def evaluate_setups(
    df: pd.DataFrame,
    symbol: str,
    tf_label: str = "15M",
    sr_out: dict | None = None,
    ob_out: dict | None = None,
    rr_target: float = 5.0,
    df_1h: pd.DataFrame | None = None,
    df_4h: pd.DataFrame | None = None,
    mtf_context: dict | None = None,
    mtf_data: dict[str, pd.DataFrame] | None = None,
) -> list[TradeSetup]:
    """
    Evaluate trading setups using the complete MTF cascade.

    Strategy hierarchy:

        4H
          ↓
        Primary bias/location
          ↓
        1H
          ↓
        Confirmation/context
          ↓
        15M
          ↓
        Primary setup
          ↓
        5M / 1M
          ↓
        Precision confirmation
          ↓
        TradeSetup

    BACKWARD-COMPATIBILITY NOTE
    ---------------------------
    The following parameters are accepted by this signature for
    compatibility with existing callers (worker.py), but are
    NOT used by this function:

        - sr_out
        - ob_out
        - mtf_context

    The S/R and OB/FVG engines are re-run internally by
    mtf_runner.run_closed_candle_5tf_cascade using df / df_1h /
    df_4h / mtf_data. Any pre-computed sr_out / ob_out passed in
    here is silently ignored.

    If you ever need the caller's sr_out / ob_out to influence
    the trade decision, that requires threading them through
    mtf_runner — it is not enough to pass them here.

    Note also that worker.py computes sr_out with
    invalidation_mode="Wick" while the cascade internally uses
    the default "Close". Because the caller's sr_out is ignored,
    the cascade effectively runs with "Close".
    """

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        return []

    if df.empty:
        return []

    if rr_target <= 0:
        return []

    frames = _prepare_mtf_frames(
        df=df,
        df_1h=df_1h,
        df_4h=df_4h,
        mtf_data=mtf_data,
    )

    if not _has_required_timeframes(
        frames
    ):
        return []

    cascade = run_closed_candle_5tf_cascade(
        frames_by_tf=frames,
        rr_target=rr_target,
    )

    setup = _cascade_to_setup(
        cascade=cascade,
        symbol=symbol,
        tf_label=tf_label,
        rr_target=rr_target,
    )

    if setup is None:
        return []

    return [setup]
