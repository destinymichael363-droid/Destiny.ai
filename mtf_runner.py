from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from ob_fvg_engine import PairedZone, _atr, run_ob_fvg_engine
from sr_engine import SRZone, run_sr_zones_engine


@dataclass
class ScalpSignal:
    direction: Literal["LONG", "SHORT"]
    confidence: str
    entry_price: float
    stop_loss: float
    take_profit: float
    rr_ratio: float
    htf_bias: str
    htf_zone_summary: str
    ltf_overlap_summary: str
    sweep_summary: str


def _closed_only(
    df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Keep the supplied data because worker.py
    already removes all forming candles.

    Do not remove another closed candle.
    """
    if (
        df is None
        or len(df) < 3
    ):
        return pd.DataFrame()

    return df.copy()


def _zones_overlap(
    low_a: float,
    high_a: float,
    low_b: float,
    high_b: float,
) -> bool:
    """Return True when two price ranges overlap."""
    return max(low_a, low_b) < min(high_a, high_b)


def _price_in_or_near_zone(
    price_low: float,
    price_high: float,
    zone_low: float,
    zone_high: float,
    tolerance: float = 0.0,
) -> bool:
    """
    Determine whether the candle touches or is within a small tolerance
    of a zone.
    """
    expanded_low = zone_low - tolerance
    expanded_high = zone_high + tolerance

    return (
        price_low <= expanded_high
        and price_high >= expanded_low
    )


def _paired_zone_bounds(
    pz: PairedZone,
) -> tuple[float, float]:
    """Return the complete OB+FVG price range."""
    return (
        min(pz.ob_low, pz.fvg_low),
        max(pz.ob_high, pz.fvg_high),
    )


def _direction_name(direction: int) -> str:
    return {
        1: "BULLISH",
        -1: "BEARISH",
        0: "NEUTRAL",
    }.get(direction, "NEUTRAL")


def _safe_last_bool(
    df: pd.DataFrame,
    column: str,
    lookback: int = 5,
) -> bool:
    """Safely check whether a boolean signal occurred recently."""
    if df.empty or column not in df.columns:
        return False

    values = df[column].tail(lookback)

    if values.empty:
        return False

    return bool(values.fillna(False).astype(bool).any())


def _recent_sweeps(
    tf: str,
    results: dict,
    lookback: int = 5,
) -> int:
    """
    Count recent liquidity sweeps.

    This is intentionally treated as supporting evidence rather than
    an independent trade trigger.
    """
    sig_df = results[tf]["ob_fvg"].get(
        "signals_df",
        pd.DataFrame(),
    )

    if sig_df.empty:
        return 0

    if "swept_liquidity_count" not in sig_df.columns:
        return 0

    values = pd.to_numeric(
        sig_df["swept_liquidity_count"].tail(lookback),
        errors="coerce",
    ).fillna(0)

    return int(values.sum())


def _get_trend(
    tf: str,
    results: dict,
) -> int:
    """
    Read the latest closed-candle directional trend.

    Returns:
        1  = bullish
        -1 = bearish
        0  = neutral/unknown
    """
    sig_df = results[tf]["ob_fvg"].get(
        "signals_df",
        pd.DataFrame(),
    )

    if sig_df.empty or "trend_direction" not in sig_df.columns:
        return 0

    value = sig_df["trend_direction"].iloc[-1]

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _get_structure_confirmation(
    tf: str,
    direction: int,
    results: dict,
    lookback: int = 5,
) -> bool:
    """
    Check for recent directional BOS/setup confirmation.

    The current OB/FVG engine exposes BOS/setup fields. These are used
    as the structural confirmation layer without inventing a separate
    CHoCH implementation.

    If the engine later exposes explicit CHoCH fields, they can be
    added here without changing the cascade architecture.
    """
    sig_df = results[tf]["ob_fvg"].get(
        "signals_df",
        pd.DataFrame(),
    )

    if sig_df.empty:
        return False

    if direction == 1:
        return (
            _safe_last_bool(
                sig_df,
                "bull_bos",
                lookback,
            )
            or
            _safe_last_bool(
                sig_df,
                "new_bull_setup",
                lookback,
            )
        )

    if direction == -1:
        return (
            _safe_last_bool(
                sig_df,
                "bear_bos",
                lookback,
            )
            or
            _safe_last_bool(
                sig_df,
                "new_bear_setup",
                lookback,
            )
        )

    return False


def _build_htf_zones(
    results: dict,
) -> list[tuple[str, SRZone]]:
    """Collect active 4H and 1H S/R zones."""
    zones: list[tuple[str, SRZone]] = []

    for tf in ("4H", "1H"):
        for zone in results[tf]["sr"].get(
            "active_zones",
            [],
        ):
            zones.append((tf, zone))

    return zones


def _build_primary_15m_overlaps(
    results: dict,
    htf_direction: int,
) -> list[dict]:
    """
    Build the primary setup layer.

    A valid primary setup requires:

        HTF zone
            ↓
        15M OB+FVG
            ↓
        15M directional alignment

    5M/1M are deliberately excluded here because they are execution
    refinement layers, not primary setup generators.
    """
    if htf_direction == 0:
        return []

    overlaps: list[dict] = []

    htf_sr_zones = _build_htf_zones(results)

    paired_1h_zones: list[PairedZone] = (
        results["1H"]["ob_fvg"].get(
            "active_zones",
            [],
        )
    )

    paired_15m_zones: list[PairedZone] = (
        results["15M"]["ob_fvg"].get(
            "active_zones",
            [],
        )
    )

    for pz_15m in paired_15m_zones:
        if pz_15m.direction != htf_direction:
            continue

        pz_low, pz_high = _paired_zone_bounds(
            pz_15m
        )

        # ---------------------------------------------------------
        # 4H / 1H S/R -> 15M OB+FVG
        # ---------------------------------------------------------
        for htf_name, srz in htf_sr_zones:
            expected_type = (
                "support"
                if htf_direction == 1
                else "resistance"
            )

            if srz.zone_type != expected_type:
                continue

            if not _zones_overlap(
                srz.bottom,
                srz.top,
                pz_low,
                pz_high,
            ):
                continue

            overlaps.append(
                {
                    "direction": htf_direction,
                    "htf_source": (
                        f"{htf_name} "
                        f"{srz.zone_type.upper()} S/R "
                        f"({srz.bottom:.5f}-"
                        f"{srz.top:.5f})"
                    ),
                    "ltf_source": (
                        f"15M "
                        f"{'BULL' if htf_direction == 1 else 'BEAR'} "
                        f"OB+FVG "
                        f"({pz_low:.5f}-"
                        f"{pz_high:.5f})"
                    ),
                    "primary_tf": "15M",
                    "ltf_tf": "15M",
                    "pz": pz_15m,
                    "primary_low": pz_low,
                    "primary_high": pz_high,
                    "priority": (
                        0 if htf_name == "4H" else 1
                    ),
                }
            )

        # ---------------------------------------------------------
        # 1H OB+FVG -> 15M OB+FVG
        # ---------------------------------------------------------
        for pz_1h in paired_1h_zones:
            if pz_1h.direction != htf_direction:
                continue

            h1_low, h1_high = _paired_zone_bounds(
                pz_1h
            )

            if not _zones_overlap(
                h1_low,
                h1_high,
                pz_low,
                pz_high,
            ):
                continue

            overlaps.append(
                {
                    "direction": htf_direction,
                    "htf_source": (
                        f"1H "
                        f"{'BULL' if htf_direction == 1 else 'BEAR'} "
                        f"OB+FVG "
                        f"({h1_low:.5f}-"
                        f"{h1_high:.5f})"
                    ),
                    "ltf_source": (
                        f"15M "
                        f"{'BULL' if htf_direction == 1 else 'BEAR'} "
                        f"OB+FVG "
                        f"({pz_low:.5f}-"
                        f"{pz_high:.5f})"
                    ),
                    "primary_tf": "15M",
                    "ltf_tf": "15M",
                    "pz": pz_15m,
                    "primary_low": pz_low,
                    "primary_high": pz_high,
                    "priority": 2,
                }
            )

    return overlaps


def _find_execution_confirmation(
    results: dict,
    primary_setup: dict,
    direction: int,
) -> dict:
    """
    Search for 5M/1M precision confirmation after the 15M setup
    has already been validated.

    Lower timeframes cannot create a setup by themselves.
    """
    primary_low = float(
        primary_setup["primary_low"]
    )
    primary_high = float(
        primary_setup["primary_high"]
    )

    confirmations: list[dict] = []

    for tf in ("5M", "1M"):
        df = results[tf]["df"]

        if df.empty:
            continue

        sig_df = results[tf]["ob_fvg"].get(
            "signals_df",
            pd.DataFrame(),
        )

        if sig_df.empty:
            continue

        last_low = float(df["Low"].iloc[-1])
        last_high = float(df["High"].iloc[-1])

        # The precision timeframe must still interact with the
        # primary 15M setup zone.
        atr_series = _atr(df, 14)

        if not atr_series.empty and not pd.isna(
            atr_series.iloc[-1]
        ):
            tolerance = float(atr_series.iloc[-1]) * 0.25
        else:
            tolerance = abs(last_high - last_low) * 0.25

        zone_interaction = _price_in_or_near_zone(
            last_low,
            last_high,
            primary_low,
            primary_high,
            tolerance=tolerance,
        )

        if not zone_interaction:
            continue

        structure_ok = _get_structure_confirmation(
            tf,
            direction,
            results,
            lookback=5,
        )

        sweep_count = _recent_sweeps(
            tf,
            results,
            lookback=8 if tf == "5M" else 10,
        )

        # A lower timeframe must either:
        #
        #   1. interact with the primary zone AND
        #      show directional structure, or
        #
        #   2. interact with the zone AND have a
        #      directional setup signal.
        #
        # It can never qualify solely because BOS occurred
        # somewhere away from the primary zone.
        if not structure_ok and sweep_count <= 0:
            continue

        confirmations.append(
            {
                "tf": tf,
                "zone_interaction": zone_interaction,
                "structure_confirmation": structure_ok,
                "sweeps": sweep_count,
                "priority": 0 if tf == "1M" else 1,
            }
        )

    if not confirmations:
        return {
            "confirmed": False,
            "tf": None,
            "structure_confirmation": False,
            "sweeps": 0,
        }

    confirmations.sort(
        key=lambda item: item["priority"]
    )

    best = confirmations[0]

    return {
        "confirmed": True,
        "tf": best["tf"],
        "structure_confirmation": bool(
            best["structure_confirmation"]
        ),
        "sweeps": int(best["sweeps"]),
    }


def run_closed_candle_5tf_cascade(
    frames_by_tf: dict[str, pd.DataFrame],
    rr_target: float = 5.0,
    atr_sl_buffer: float = 0.25,
) -> dict:
    """
    Closed-candle multi-timeframe trading cascade.

    Architecture:

        4H
         ↓
        PRIMARY BIAS / LOCATION
         ↓
        1H
         ↓
        CONFIRMATION / CONTEXT
         ↓
        15M
         ↓
        PRIMARY SETUP
         ↓
        5M
         ↓
        PRECISION CONFIRMATION
         ↓
        1M
         ↓
        EXECUTION

    Expected keys:

        4H
        1H
        15M
        5M
        1M

    The function remains deterministic and rule-based.
    No machine learning is introduced here.
    """
    tf_minutes_map = {
        "4H": 240,
        "1H": 60,
        "15M": 15,
        "5M": 5,
        "1M": 1,
    }

    timeframe_order = [
        "4H",
        "1H",
        "15M",
        "5M",
        "1M",
    ]

    results: dict[str, dict] = {}
    # =============================================================
    # 1. Build each timeframe from closed candles only
    # =============================================================
    for tf in timeframe_order:
        closed_df = _closed_only(
            frames_by_tf.get(tf)
        )

        if closed_df.empty:
            results[tf] = {
                "df": closed_df,
                "sr": {
                    "active_zones": [],
                    "minor_zones": [],
                    "all_zones": [],
                },
                "ob_fvg": {
                    "active_zones": [],
                    "signals_df": pd.DataFrame(),
                },
            }
            continue

        sr_res = run_sr_zones_engine(
            closed_df,
            tf_label=tf,
        )

        ob_res = run_ob_fvg_engine(
            closed_df,
            timeframe_minutes=tf_minutes_map[tf],
        )

        results[tf] = {
            "df": closed_df,
            "sr": sr_res,
            "ob_fvg": ob_res,
        }

    # =============================================================
    # 2. Read directional trends
    # =============================================================
    tf_trends = {
        tf: _get_trend(tf, results)
        for tf in timeframe_order
            }

    t_4h = tf_trends["4H"]
    t_1h = tf_trends["1H"]
    t_15m = tf_trends["15M"]
    t_5m = tf_trends["5M"]
    t_1m = tf_trends["1M"]

    # =============================================================
    # 3. 4H MUST establish the primary bias
    # =============================================================
    #
    # 4H is no longer allowed to be neutral while 1H creates
    # a trade direction.
    #
    # 4H = primary authority.
    #
    htf_dir = 0

    if t_4h == 1:
        # 1H can be bullish or temporarily neutral.
        # A bearish 1H blocks the setup.
        if t_1h in (1, 0):
            htf_dir = 1

    elif t_4h == -1:
        # 1H can be bearish or temporarily neutral.
        # A bullish 1H blocks the setup.
        if t_1h in (-1, 0):
            htf_dir = -1

    # =============================================================
    # 4. Recent liquidity sweeps
    # =============================================================
    sweeps_15m = _recent_sweeps(
        "15M",
        results,
        lookback=6,
    )

    sweeps_5m = _recent_sweeps(
        "5M",
        results,
        lookback=8,
    )

    sweeps_1m = _recent_sweeps(
        "1M",
        results,
        lookback=10,
    )

    total_recent_sweeps = (
        sweeps_15m
        + sweeps_5m
        + sweeps_1m
    )

    # =============================================================
    # 5. 15M is the PRIMARY setup layer
    # =============================================================
    primary_overlaps: list[dict] = []

    if htf_dir != 0:
        primary_overlaps = (
            _build_primary_15m_overlaps(
                results,
                htf_dir,
            )
        )

    # =============================================================
    # 6. 15M structural confirmation
    # =============================================================
    m15_structure_ok = False

    if htf_dir != 0:
        m15_structure_ok = (
            _get_structure_confirmation(
                "15M",
                htf_dir,
                results,
                lookback=5,
            )
        )

    # =============================================================
    # 7. Select best PRIMARY setup
    # =============================================================
    primary_setup: dict | None = None

    if primary_overlaps:
        primary_overlaps.sort(
            key=lambda item: (
                item["priority"],
                -(
                    item["primary_high"]
                    - item["primary_low"]
                ),
            )
        )

        primary_setup = primary_overlaps[0]

    # =============================================================
    # 8. Require the complete HTF -> 15M chain
    # =============================================================
    setup_valid = bool(
        htf_dir != 0
        and primary_setup is not None
        and m15_structure_ok
    )

    # =============================================================
    # 9. 5M / 1M precision confirmation
    # =============================================================
    execution_confirmation = {
        "confirmed": False,
        "tf": None,
        "structure_confirmation": False,
        "sweeps": 0,
    }

    if setup_valid and primary_setup is not None:
        execution_confirmation = (
            _find_execution_confirmation(
                results,
                primary_setup,
                htf_dir,
            )
        )

    # =============================================================
    # 10. Build compatibility overlaps output
    # =============================================================
    #
    # Keep the original "overlaps" output so existing downstream
    # code does not break.
    #
    # Primary 15M overlaps are returned first.
    #
    overlaps = list(primary_overlaps)

    # =============================================================
    # 11. Execute only after the complete chain is valid
    # =============================================================
    df_1m = results["1M"]["df"]

    sig_1m = results["1M"]["ob_fvg"].get(
        "signals_df",
        pd.DataFrame(),
    )

    scalp_signal: ScalpSignal | None = None

    if (
        setup_valid
        and primary_setup is not None
        and execution_confirmation["confirmed"]
        and not df_1m.empty
    ):
        last_close_1m = float(
            df_1m["Close"].iloc[-1]
        )

        last_low_1m = float(
            df_1m["Low"].iloc[-1]
        )

        last_high_1m = float(
            df_1m["High"].iloc[-1]
        )

        atr_1m_series = _atr(
            df_1m,
            14,
        )

        if (
            not atr_1m_series.empty
            and not pd.isna(
                atr_1m_series.iloc[-1]
            )
        ):
            atr_1m = float(
                atr_1m_series.iloc[-1]
            )
        else:
            atr_1m = abs(
                last_high_1m
                - last_low_1m
            )

        # Prevent zero-distance risk.
        atr_1m = max(
            atr_1m,
            abs(last_high_1m - last_low_1m),
            1e-12,
        )

        # ---------------------------------------------------------
        # Primary 15M zone controls invalidation.
        # ---------------------------------------------------------
        primary_low = float(
            primary_setup["primary_low"]
        )

        primary_high = float(
            primary_setup["primary_high"]
        )

        if htf_dir == 1:
            sl = (
                min(
                    primary_low,
                    last_low_1m,
                )
                - (
                    atr_1m
                    * atr_sl_buffer
                )
            )

            risk_dist = (
                last_close_1m
                - sl
            )

            risk_dist = max(
                risk_dist,
                atr_1m * 0.5,
            )

            sl = (
                last_close_1m
                - risk_dist
            )

            tp = (
                last_close_1m
                + (
                    risk_dist
                    * rr_target
                )
            )

            side: Literal[
                "LONG",
                "SHORT",
            ] = "LONG"

        else:
            sl = (
                max(
                    primary_high,
                    last_high_1m,
                )
                + (
                    atr_1m
                    * atr_sl_buffer
                )
            )

            risk_dist = (
                sl
                - last_close_1m
            )

            risk_dist = max(
                risk_dist,
                atr_1m * 0.5,
            )

            sl = (
                last_close_1m
                + risk_dist
            )

            tp = (
                last_close_1m
                - (
                    risk_dist
                    * rr_target
                )
            )

            side = "SHORT"

        # ---------------------------------------------------------
        # Final risk validation
        # ---------------------------------------------------------
        valid_risk = (
            risk_dist > 0
            and tp > last_close_1m
            if side == "LONG"
            else (
                risk_dist > 0
                and tp < last_close_1m
            )
        )

        calculated_rr = (
            abs(tp - last_close_1m)
            / risk_dist
            if risk_dist > 0
            else 0.0
        )

        valid_rr = (
            calculated_rr >= rr_target * 0.99
        )

        if valid_risk and valid_rr:
            m15_sweep = sweeps_15m > 0
            lower_sweep = (
                sweeps_5m > 0
                or sweeps_1m > 0
            )

            ltf_confirmed = (
                execution_confirmation[
                    "structure_confirmation"
                ]
            )

            if (
                m15_sweep
                and ltf_confirmed
            ):
                confidence = (
                    "HIGH "
                    "(15M Sweep + HTF/15M "
                    "Overlap + LTF Confirmed)"
                )

            elif m15_sweep:
                confidence = (
                    "HIGH "
                    "(15M Sweep + HTF/15M "
                    "Overlap)"
                )

            elif ltf_confirmed:
                confidence = (
                    "STANDARD "
                    "(HTF/15M Overlap + "
                    "LTF Confirmed)"
                )

            elif lower_sweep:
                confidence = (
                    "STANDARD "
                    "(HTF/15M Overlap + "
                    "LTF Sweep)"
                )

            else:
                confidence = (
                    "STANDARD "
                    "(HTF/15M Overlap)"
                )

            execution_tf = (
                execution_confirmation["tf"]
                or "LTF"
            )

            scalp_signal = ScalpSignal(
                direction=side,
                confidence=confidence,
                entry_price=round(
                    last_close_1m,
                    5,
                ),
                stop_loss=round(
                    sl,
                    5,
                ),
                take_profit=round(
                    tp,
                    5,
                ),
                rr_ratio=round(
                    calculated_rr,
                    3,
                ),
                htf_bias=_direction_name(
                    htf_dir
                ),
                htf_zone_summary=(
                    primary_setup[
                        "htf_source"
                    ]
                ),
                ltf_overlap_summary=(
                    primary_setup[
                        "ltf_source"
                    ]
                    + f" | "
                    f"15M structure confirmed"
                    + f" | "
                    f"{execution_tf} execution"
                ),
                sweep_summary=(
                    f"15M:{sweeps_15m} | "
                    f"5M:{sweeps_5m} | "
                    f"1M:{sweeps_1m} "
                    f"liquidity sweeps"
                ),
            )

    # =============================================================
    # 12. Return complete cascade state
    # =============================================================
    return {
        "tf_trends": {
            "4H": t_4h,
            "1H": t_1h,
            "15M": t_15m,
            "5M": t_5m,
            "1M": t_1m,
        },

        "htf_bias": htf_dir,

        "htf_bias_label": _direction_name(
            htf_dir
        ),

        "primary_setup": {
            "valid": setup_valid,
            "direction": htf_dir,
            "overlap": primary_setup,
            "structure_confirmation": (
                m15_structure_ok
            ),
            "sweeps_15m": sweeps_15m,
        },

        "execution_confirmation": (
            execution_confirmation
        ),

        "overlaps": overlaps,

        "recent_sweeps": {
            "15M": sweeps_15m,
            "5M": sweeps_5m,
            "1M": sweeps_1m,
            "total": total_recent_sweeps,
        },

        "scalp_signal": scalp_signal,

        "per_tf": results,
    }
