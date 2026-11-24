from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TradeSetup:
    symbol: str
    tf_label: str
    direction: str      # "LONG" or "SHORT"
    entry: float
    sl: float
    tp: float
    rr: float
    confidence: str     # "HIGH", "MEDIUM", or "LOW"
    reason: str


# =====================================================================
# BASIC HELPERS
# =====================================================================

def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        value_f = float(value)
        if np.isfinite(value_f):
            return value_f
    except (TypeError, ValueError):
        pass

    return default


def _zone_bounds(
    zone: object,
) -> tuple[float, float, str] | None:
    """
    Extract a usable low/high range from either:

        - dictionary-based zones
        - SRZone-like objects
        - OB/FVG-like objects

    This preserves compatibility with the existing engines.
    """
    if isinstance(zone, dict):
        low_v = zone.get(
            "bottom",
            zone.get(
                "overlap_low",
                zone.get(
                    "ob_low",
                    zone.get("low"),
                ),
            ),
        )

        high_v = zone.get(
            "top",
            zone.get(
                "overlap_high",
                zone.get(
                    "ob_high",
                    zone.get("high"),
                ),
            ),
        )

        z_type = str(
            zone.get(
                "zone_type",
                zone.get(
                    "kind",
                    zone.get(
                        "direction",
                        "",
                    ),
                ),
            )
        ).lower()

    else:
        low_v = getattr(
            zone,
            "bottom",
            getattr(
                zone,
                "low",
                getattr(
                    zone,
                    "ob_low",
                    None,
                ),
            ),
        )

        high_v = getattr(
            zone,
            "top",
            getattr(
                zone,
                "high",
                getattr(
                    zone,
                    "ob_high",
                    None,
                ),
            ),
        )

        z_type = str(
            getattr(
                zone,
                "zone_type",
                getattr(
                    zone,
                    "kind",
                    getattr(
                        zone,
                        "direction",
                        "",
                    ),
                ),
            )
        ).lower()

    if low_v is None or high_v is None:
        return None

    try:
        bottom = float(low_v)
        top = float(high_v)
    except (TypeError, ValueError):
        return None

    if not (
        np.isfinite(bottom)
        and np.isfinite(top)
    ):
        return None

    if bottom > top:
        bottom, top = top, bottom

    return bottom, top, z_type


def _atr_proxy(
    df: pd.DataFrame,
    length: int = 14,
) -> float:
    """
    Lightweight ATR-like volatility estimate.

    This is retained for compatibility with the original strategy
    engine and is only used by the legacy fallback path.
    """
    if df is None or df.empty:
        return 0.0

    required = {"High", "Low"}

    if not required.issubset(df.columns):
        return 0.0

    high = pd.to_numeric(
        df["High"],
        errors="coerce",
    )

    low = pd.to_numeric(
        df["Low"],
        errors="coerce",
    )

    ranges = (
        high - low
    ).abs().dropna().tail(length)

    if ranges.empty:
        return 0.0

    value = float(ranges.mean())

    if not np.isfinite(value):
        return 0.0

    return max(value, 0.0)


def _normalise_direction(
    direction: object,
) -> str:
    """
    Convert numeric/string direction values into LONG/SHORT/NEUTRAL.
    """
    if isinstance(direction, str):
        value = direction.strip().upper()

        if value in {
            "LONG",
            "BULL",
            "BULLISH",
            "BUY",
            "1",
            "+1",
        }:
            return "LONG"

        if value in {
            "SHORT",
            "BEAR",
            "BEARISH",
            "SELL",
            "-1",
        }:
            return "SHORT"

        return "NEUTRAL"

    try:
        value = int(direction)

        if value > 0:
            return "LONG"

        if value < 0:
            return "SHORT"

    except (TypeError, ValueError):
        pass

    return "NEUTRAL"


def _cascade_from_context(
    mtf_context: dict | None,
) -> dict | None:
    """
    Locate the corrected 5TF cascade inside mtf_context.

    worker.py stores the result under:

        mtf_context["cascade"]

    A direct cascade dictionary is also accepted for compatibility.
    """
    if not isinstance(mtf_context, dict):
        return None

    cascade = mtf_context.get("cascade")

    if isinstance(cascade, dict):
        return cascade

    # Compatibility if a caller passes the cascade directly.
    if (
        "tf_trends" in mtf_context
        and "primary_setup" in mtf_context
        and "scalp_signal" in mtf_context
    ):
        return mtf_context

    return None


def _signal_confidence(
    cascade_signal: object,
) -> str:
    """
    Convert the detailed MTF runner confidence labels into the
    strategy engine's existing HIGH/MEDIUM/LOW interface.
    """
    confidence = str(
        getattr(
            cascade_signal,
            "confidence",
            "",
        )
    ).upper()

    if confidence.startswith("HIGH"):
        return "HIGH"

    if confidence.startswith("STANDARD"):
        return "MEDIUM"

    if confidence.startswith("MEDIUM"):
        return "MEDIUM"

    return "LOW"


# =====================================================================
# PRIMARY MTF CASCADE PATH
# =====================================================================

def _evaluate_mtf_cascade(
    cascade: dict,
    symbol: str,
    tf_label: str,
) -> list[TradeSetup]:
    """
    Convert the corrected mtf_runner.py result into the existing
    TradeSetup interface.

    The cascade is authoritative.

    Architecture:

        4H
         ↓
        1H
         ↓
        15M
         ↓
        5M
         ↓
        1M
    """
    if not isinstance(cascade, dict):
        return []

    primary_setup = cascade.get(
        "primary_setup"
    )

    if not isinstance(primary_setup, dict):
        return []

    # The MTF runner explicitly decides whether the complete chain
    # is valid.
    if not bool(
        primary_setup.get("valid", False)
    ):
        return []

    direction = _normalise_direction(
        primary_setup.get("direction", 0)
    )

    if direction == "NEUTRAL":
        return []

    scalp_signal = cascade.get(
        "scalp_signal"
    )

    # The runner only creates scalp_signal after:
    #
    #   HTF bias
    #   +
    #   15M setup
    #   +
    #   15M structure
    #   +
    #   5M/1M confirmation
    #   +
    #   valid risk/reward
    #
    if scalp_signal is None:
        return []

    entry = _safe_float(
        getattr(
            scalp_signal,
            "entry_price",
            None,
        )
    )

    sl = _safe_float(
        getattr(
            scalp_signal,
            "stop_loss",
            None,
        )
    )

    tp = _safe_float(
        getattr(
            scalp_signal,
            "take_profit",
            None,
        )
    )

    rr = _safe_float(
        getattr(
            scalp_signal,
            "rr_ratio",
            None,
        )
    )

    if (
        entry <= 0
        or sl <= 0
        or tp <= 0
        or rr <= 0
    ):
        return []

    if direction == "LONG":
        if not (
            sl < entry < tp
        ):
            return []

    elif direction == "SHORT":
        if not (
            tp < entry < sl
        ):
            return []

    # -------------------------------------------------------------
    # Extract cascade metadata for a transparent reason string.
    # -------------------------------------------------------------

    htf_bias_label = str(
        cascade.get(
            "htf_bias_label",
            getattr(
                scalp_signal,
                "htf_bias",
                "",
            ),
        )
    )

    execution_confirmation = cascade.get(
        "execution_confirmation",
        {},
    )

    if not isinstance(
        execution_confirmation,
        dict,
    ):
        execution_confirmation = {}

    execution_tf = (
        execution_confirmation.get(
            "tf"
        )
        or "LTF"
    )

    structure_confirmed = bool(
        primary_setup.get(
            "structure_confirmation",
            False,
        )
    )

    sweeps = cascade.get(
        "recent_sweeps",
        {},
    )

    if not isinstance(sweeps, dict):
        sweeps = {}

    sweep_15m = int(
        _safe_float(
            sweeps.get("15M", 0)
        )
    )

    sweep_5m = int(
        _safe_float(
            sweeps.get("5M", 0)
        )
    )

    sweep_1m = int(
        _safe_float(
            sweeps.get("1M", 0)
        )
    )

    confidence = _signal_confidence(
        scalp_signal
    )

    # -------------------------------------------------------------
    # Preserve the detailed reason from mtf_runner.py while adding
    # enough context for logs/UI.
    # -------------------------------------------------------------

    runner_confidence = str(
        getattr(
            scalp_signal,
            "confidence",
            "",
        )
    )

    htf_summary = str(
        getattr(
            scalp_signal,
            "htf_zone_summary",
            "",
        )
    )

    ltf_summary = str(
        getattr(
            scalp_signal,
            "ltf_overlap_summary",
            "",
        )
    )

    reason_parts = [
        "DESTINY MTF cascade",
        f"4H/1H bias={htf_bias_label}",
        "15M primary setup",
    ]

    if structure_confirmed:
        reason_parts.append(
            "15M structure confirmed"
        )

    if execution_tf:
        reason_parts.append(
            f"{execution_tf} execution confirmation"
        )

    if sweep_15m > 0:
        reason_parts.append(
            f"15M sweep={sweep_15m}"
        )

    if sweep_5m > 0:
        reason_parts.append(
            f"5M sweep={sweep_5m}"
        )

    if sweep_1m > 0:
        reason_parts.append(
            f"1M sweep={sweep_1m}"
        )

    if runner_confidence:
        reason_parts.append(
            runner_confidence
        )

    if htf_summary:
        reason_parts.append(
            htf_summary
        )

    if ltf_summary:
        reason_parts.append(
            ltf_summary
        )

    return [
        TradeSetup(
            symbol=symbol,
            tf_label=tf_label,
            direction=direction,
            entry=round(entry, 5),
            sl=round(sl, 5),
            tp=round(tp, 5),
            rr=round(rr, 3),
            confidence=confidence,
            reason=" | ".join(reason_parts),
        )
    ]


# =====================================================================
# LEGACY COMPATIBILITY PATH
# =====================================================================

def _evaluate_legacy_setups(
    df: pd.DataFrame,
    symbol: str,
    tf_label: str,
    sr_out: dict | None,
    ob_out: dict | None,
    rr_target: float,
    max_setups: int,
) -> list[TradeSetup]:
    """
    Preserve the original strategy logic as a fallback.

    This path is used only when the corrected MTF cascade is not
    available.

    It is intentionally not allowed to compete with the authoritative
    5TF cascade when the cascade has produced a result.
    """
    if (
        df is None
        or df.empty
        or len(df) < 15
    ):
        return []

    required = {
        "Open",
        "High",
        "Low",
        "Close",
    }

    if not required.issubset(df.columns):
        return []

    data = df.copy()

    for column in required:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=list(required)
    )

    if len(data) < 15:
        return []

    last_row = data.iloc[-1]

    price = _safe_float(
        last_row["Close"]
    )

    if price <= 0:
        return []

    atr_proxy = _atr_proxy(
        data,
        14,
    )

    buffer = max(
        atr_proxy * 0.35,
        abs(price) * 0.0005,
    )

    setups: list[TradeSetup] = []

    active_sr = (
        (sr_out or {}).get(
            "active_zones",
            [],
        )
        or []
    )

    overlaps = (
        (ob_out or {}).get(
            "overlaps",
            [],
        )
        or []
    )

    # -------------------------------------------------------------
    # Original S/R setup detection.
    # -------------------------------------------------------------

    for zone in active_sr[-4:]:
        bounds = _zone_bounds(zone)

        if not bounds:
            continue

        bot, top, z_type = bounds

        is_bull_zone = (
            "sup" in z_type
            or "demand" in z_type
            or "bull" in z_type
        )

        is_bear_zone = (
            "res" in z_type
            or "supply" in z_type
            or "bear" in z_type
        )

        bullish_candle = (
            _safe_float(
                last_row["Close"]
            )
            >
            _safe_float(
                last_row["Open"]
            )
        )

        bearish_candle = (
            _safe_float(
                last_row["Close"]
            )
            <
            _safe_float(
                last_row["Open"]
            )
        )

        if (
            is_bull_zone
            and price >= bot * 0.998
            and bullish_candle
        ):
            sl = round(
                bot - buffer,
                5,
            )

            risk = max(
                price - sl,
                buffer,
            )

            tp = round(
                price
                + (
                    risk
                    * float(rr_target)
                ),
                5,
            )

            setups.append(
                TradeSetup(
                    symbol=symbol,
                    tf_label=tf_label,
                    direction="LONG",
                    entry=round(
                        price,
                        5,
                    ),
                    sl=sl,
                    tp=tp,
                    rr=float(rr_target),
                    confidence="MEDIUM",
                    reason=(
                        f"Bullish rejection at "
                        f"{tf_label} support zone "
                        f"[{bot:.5f}–{top:.5f}]"
                    ),
                )
            )

            break

        if (
            is_bear_zone
            and price <= top * 1.002
            and bearish_candle
        ):
            sl = round(
                top + buffer,
                5,
            )

            risk = max(
                sl - price,
                buffer,
            )

            tp = round(
                price
                - (
                    risk
                    * float(rr_target)
                ),
                5,
            )

            setups.append(
                TradeSetup(
                    symbol=symbol,
                    tf_label=tf_label,
                    direction="SHORT",
                    entry=round(
                        price,
                        5,
                    ),
                    sl=sl,
                    tp=tp,
                    rr=float(rr_target),
                    confidence="MEDIUM",
                    reason=(
                        f"Bearish rejection at "
                        f"{tf_label} resistance zone "
                        f"[{bot:.5f}–{top:.5f}]"
                    ),
                )
            )

            break

    # -------------------------------------------------------------
    # Original OB/FVG confidence enhancement.
    # -------------------------------------------------------------

    for ov in overlaps[-3:]:
        bounds = _zone_bounds(ov)

        if not bounds:
            continue

        bot, top, z_type = bounds

        for setup in setups:
            if (
                setup.direction == "LONG"
                and (
                    (
                        bot
                        <= setup.entry
                        <= top * 1.01
                    )
                    or "1" in z_type
                    or "bull" in z_type
                )
            ):
                setup.confidence = "HIGH"

                setup.reason += (
                    " + OB/FVG overlap confluence"
                )

            elif (
                setup.direction == "SHORT"
                and (
                    (
                        bot * 0.99
                        <= setup.entry
                        <= top
                    )
                    or "-1" in z_type
                    or "bear" in z_type
                )
            ):
                setup.confidence = "HIGH"

                setup.reason += (
                    " + OB/FVG overlap confluence"
                )

    return setups[:max_setups]


# =====================================================================
# PUBLIC STRATEGY API
# =====================================================================

def evaluate_setups(
    df: pd.DataFrame,
    symbol: str = "ASSET",
    tf_label: str = "15M",
    sr_out: dict | None = None,
    ob_out: dict | None = None,
    rr_target: float = 5.0,
    max_setups: int = 3,

    # -------------------------------------------------------------
    # MTF inputs
    #
    # These are optional so older callers remain compatible.
    # -------------------------------------------------------------
    df_1h: pd.DataFrame | None = None,
    df_4h: pd.DataFrame | None = None,
    df_5m: pd.DataFrame | None = None,
    df_1m: pd.DataFrame | None = None,
    mtf_context: dict | None = None,
) -> list[TradeSetup]:
    """
    Evaluate trading setups.

    PRIMARY MODE
    ------------
    When worker.py provides the corrected MTF cascade:

        4H -> 1H -> 15M -> 5M -> 1M

    the cascade is authoritative.

    FALLBACK MODE
    -------------
    If the cascade is unavailable, the original S/R + OB/FVG
    strategy remains available for compatibility.

    This design prevents the old strategy from accidentally creating
    a weaker trade when the new MTF architecture is available.
    """

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or df.empty
    ):
        return []

    # =============================================================
    # 1. Try the corrected MTF cascade first.
    # =============================================================

    cascade = _cascade_from_context(
        mtf_context
    )

    if cascade is not None:
        return _evaluate_mtf_cascade(
            cascade=cascade,
            symbol=symbol,
            tf_label=tf_label,
        )

    # =============================================================
    # 2. Compatibility fallback.
    #
    # Only reached if the MTF cascade is genuinely unavailable.
    # =============================================================

    return _evaluate_legacy_setups(
        df=df,
        symbol=symbol,
        tf_label=tf_label,
        sr_out=sr_out,
        ob_out=ob_out,
        rr_target=rr_target,
        max_setups=max_setups,
)
