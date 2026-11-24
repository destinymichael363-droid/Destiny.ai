from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from ob_fvg_engine import (
    PairedZone,
    _atr,
    run_ob_fvg_engine,
)
from sr_engine import (
    SRZone,
    run_sr_zones_engine,
)


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


# ================================================================
# BASIC HELPERS
# ================================================================

def _closed_only(
    df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Remove the newest potentially-forming candle.

    The cascade therefore works only from completed candles.

    This is especially important when the caller passes data from
    a live market feed where the final row may still be moving.
    """

    if (
        df is None
        or not isinstance(df, pd.DataFrame)
        or len(df) < 3
    ):
        return pd.DataFrame()

    return df.iloc[:-1].copy()


def _zones_overlap(
    low_a: float,
    high_a: float,
    low_b: float,
    high_b: float,
) -> bool:
    """
    True only when two price intervals genuinely overlap.
    """

    return max(
        low_a,
        low_b,
    ) < min(
        high_a,
        high_b,
    )


def _paired_zone_bounds(
    pz: PairedZone,
) -> tuple[float, float]:
    """
    Return the complete OB/FVG union.

    This preserves the original behavior where the execution zone
    covers both the OB and FVG rather than selecting only one.
    """

    return (
        min(
            pz.ob_low,
            pz.fvg_low,
        ),
        max(
            pz.ob_high,
            pz.fvg_high,
        ),
    )


def _direction_label(
    direction: int,
) -> str:
    return {
        1: "BULLISH",
        -1: "BEARISH",
        0: "NEUTRAL",
    }.get(
        int(direction),
        "NEUTRAL",
    )


def _latest_signal_row(
    result: dict,
) -> dict:
    """
    Safely retrieve the latest structural signal.
    """

    sig_df = result.get(
        "signals_df",
        pd.DataFrame(),
    )

    if (
        sig_df is None
        or sig_df.empty
    ):
        return {}

    row = sig_df.iloc[-1]

    return row.to_dict()


def _recent_column_any(
    result: dict,
    column: str,
    lookback: int,
) -> bool:
    """
    Safely determine whether a boolean/event column occurred recently.
    """

    sig_df = result.get(
        "signals_df",
        pd.DataFrame(),
    )

    if (
        sig_df is None
        or sig_df.empty
        or column not in sig_df.columns
    ):
        return False

    values = (
        sig_df[column]
        .tail(max(int(lookback), 1))
        .fillna(False)
    )

    return bool(
        values.astype(bool).any()
    )


def _recent_sweeps(
    result: dict,
    lookback: int = 5,
) -> int:
    """
    Count recent liquidity sweeps.

    Uses the corrected OB/FVG engine's sweep records.
    """

    sig_df = result.get(
        "signals_df",
        pd.DataFrame(),
    )

    if (
        sig_df is None
        or sig_df.empty
        or "swept_liquidity_count"
        not in sig_df.columns
    ):
        return 0

    values = pd.to_numeric(
        sig_df[
            "swept_liquidity_count"
        ].tail(
            max(int(lookback), 1)
        ),
        errors="coerce",
    ).fillna(0)

    return int(
        values.sum()
    )


def _recent_directional_sweeps(
    result: dict,
    direction: int,
    lookback: int,
) -> int:
    """
    Count sweeps relevant to the intended direction.

    LONG:
        SSL sweeps are the meaningful liquidity event.

    SHORT:
        BSL sweeps are the meaningful liquidity event.
    """

    sig_df = result.get(
        "signals_df",
        pd.DataFrame(),
    )

    if (
        sig_df is None
        or sig_df.empty
    ):
        return 0

    if direction == 1:
        column = "swept_ssl_count"

    elif direction == -1:
        column = "swept_bsl_count"

    else:
        return 0

    if column not in sig_df.columns:
        return 0

    values = pd.to_numeric(
        sig_df[column].tail(
            max(int(lookback), 1)
        ),
        errors="coerce",
    ).fillna(0)

    return int(
        values.sum()
    )


# ================================================================
# MAIN 5-TIMEFRAME CASCADE
# ================================================================

def run_closed_candle_5tf_cascade(
    frames_by_tf: dict[str, pd.DataFrame],
    rr_target: float = 5.0,
    atr_sl_buffer: float = 0.25,
) -> dict:
    """
    DESTINY AI closed-candle multi-timeframe cascade.

    Hierarchy:

        4H
          ↓
        primary location + bias

        1H
          ↓
        HTF confirmation/context

        15M
          ↓
        PRIMARY SETUP CONFIRMATION
        BOS / CHoCH
        liquidity sweep
        displacement
        OB + FVG

        5M
          ↓
        refinement

        1M
          ↓
        precision entry

    IMPORTANT:

    5M and 1M are NOT allowed to create a trade when the
    15M setup is invalid.

    They can only refine a setup that has already passed the
    higher-level conditions.
    """

    # ============================================================
    # 1. SANITIZE PARAMETERS
    # ============================================================

    rr_target = max(
        float(rr_target),
        0.1,
    )

    atr_sl_buffer = max(
        float(atr_sl_buffer),
        0.0,
    )

    tf_order = [
        "4H",
        "1H",
        "15M",
        "5M",
        "1M",
    ]

    tf_minutes_map = {
        "4H": 240,
        "1H": 60,
        "15M": 15,
        "5M": 5,
        "1M": 1,
    }

    results: dict[
        str,
        dict,
    ] = {}

    # ============================================================
    # 2. RUN EACH TIMEFRAME
    # ============================================================

    for tf in tf_order:

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
                    "active_liquidity": [],
                    "signals_df": (
                        pd.DataFrame()
                    ),
                },
            }

            continue

        # --------------------------------------------------------
        # S/R engine
        # --------------------------------------------------------

        sr_res = run_sr_zones_engine(
            closed_df,
            tf_label=tf,
        )

        # --------------------------------------------------------
        # OB/FVG engine
        # --------------------------------------------------------

        ob_res = run_ob_fvg_engine(
            closed_df,
            timeframe_minutes=(
                tf_minutes_map[tf]
            ),
        )

        results[tf] = {
            "df": closed_df,
            "sr": sr_res,
            "ob_fvg": ob_res,
        }

    # ============================================================
    # 3. TREND HELPERS
    # ============================================================

    def _get_trend(
        tf: str,
    ) -> int:

        sig_df = results[tf][
            "ob_fvg"
        ].get(
            "signals_df",
            pd.DataFrame(),
        )

        if (
            sig_df is None
            or sig_df.empty
            or "trend_direction"
            not in sig_df.columns
        ):
            return 0

        value = pd.to_numeric(
            sig_df[
                "trend_direction"
            ].iloc[-1],
            errors="coerce",
        )

        if pd.isna(value):
            return 0

        return int(value)

    t_4h = _get_trend("4H")
    t_1h = _get_trend("1H")
    t_15m = _get_trend("15M")
    t_5m = _get_trend("5M")
    t_1m = _get_trend("1M")

    # ============================================================
    # 4. PRIMARY 4H BIAS
    # ============================================================
    #
    # IMPORTANT CHANGE:
    #
    # 4H is now the primary directional authority.
    #
    # We do NOT allow a bullish 1H trend to override a bearish
    # 4H trend.
    #
    # This matches:
    #
    #       4H = location + primary bias
    #       1H = confirmation
    #
    # ============================================================

    if t_4h == 1:
        htf_dir = (
            1
            if t_1h in (1, 0)
            else 0
        )

    elif t_4h == -1:
        htf_dir = (
            -1
            if t_1h in (-1, 0)
            else 0
        )

    else:
        # No reliable 4H direction means no directional setup.
        htf_dir = 0

    # ============================================================
    # 5. HTF S/R
    # ============================================================

    htf_sr_zones: list[
        tuple[str, SRZone]
    ] = (
        [
            (
                "4H",
                z,
            )
            for z in results["4H"][
                "sr"
            ].get(
                "active_zones",
                [],
            )
        ]
        +
        [
            (
                "1H",
                z,
            )
            for z in results["1H"][
                "sr"
            ].get(
                "active_zones",
                [],
            )
        ]
    )

    # ============================================================
    # 6. HTF OB/FVG
    # ============================================================

    h1_paired_zones: list[
        PairedZone
    ] = results["1H"][
        "ob_fvg"
    ].get(
        "active_zones",
        [],
    )

    h4_paired_zones: list[
        PairedZone
    ] = results["4H"][
        "ob_fvg"
    ].get(
        "active_zones",
        [],
    )

    # ============================================================
    # 7. FIND 15M PRIMARY SETUPS
    # ============================================================
    #
    # This is the central correction.
    #
    # We no longer allow a 5M/1M zone to become the primary setup.
    #
    # A candidate must first exist on 15M.
    #
    # ============================================================

    primary_15m_candidates: list[
        dict
    ] = []

    df_15m = results["15M"][
        "df"
    ]

    ob_15m = results["15M"][
        "ob_fvg"
    ]

    sig_15m = ob_15m.get(
        "signals_df",
        pd.DataFrame(),
    )

    latest_15m = _latest_signal_row(
        ob_15m
    )

    # Recent structural events.
    bull_15m_bos = _recent_column_any(
        ob_15m,
        "bull_bos",
        6,
    )

    bear_15m_bos = _recent_column_any(
        ob_15m,
        "bear_bos",
        6,
    )

    bull_15m_choch = (
        _recent_column_any(
            ob_15m,
            "choch_bull",
            6,
        )
        or
        _recent_column_any(
            ob_15m,
            "bull_choch",
            6,
        )
    )

    bear_15m_choch = (
        _recent_column_any(
            ob_15m,
            "choch_bear",
            6,
        )
        or
        _recent_column_any(
            ob_15m,
            "bear_choch",
            6,
        )
    )

    bull_15m_structure = (
        bull_15m_bos
        or bull_15m_choch
    )

    bear_15m_structure = (
        bear_15m_bos
        or bear_15m_choch
    )

    # Directional liquidity sweep.
    bull_15m_sweep = (
        _recent_directional_sweeps(
            ob_15m,
            1,
            8,
        )
        > 0
    )

    bear_15m_sweep = (
        _recent_directional_sweeps(
            ob_15m,
            -1,
            8,
        )
        > 0
    )

    # ------------------------------------------------------------
    # 15M paired zones
    # ------------------------------------------------------------

    paired_15m_zones: list[
        PairedZone
    ] = ob_15m.get(
        "active_zones",
        [],
    )

    for pz in paired_15m_zones:

        if htf_dir == 0:
            continue

        if pz.direction != htf_dir:
            continue

        pz_low, pz_high = (
            _paired_zone_bounds(pz)
        )

        # --------------------------------------------------------
        # Check whether the 15M zone is actually located inside
        # an HTF S/R zone.
        # --------------------------------------------------------

        matching_htf_sr: list[
            str
        ] = []

        expected_sr_type = (
            "support"
            if pz.direction == 1
            else "resistance"
        )

        for htf_name, srz in htf_sr_zones:

            if (
                srz.zone_type
                != expected_sr_type
            ):
                continue

            if _zones_overlap(
                srz.bottom,
                srz.top,
                pz_low,
                pz_high,
            ):
                matching_htf_sr.append(
                    (
                        f"{htf_name} "
                        f"{srz.zone_type.upper()} "
                        f"S/R "
                        f"({srz.bottom:.5f}-"
                        f"{srz.top:.5f})"
                    )
                )

        # --------------------------------------------------------
        # Check 15M zone against 1H OB/FVG.
        # --------------------------------------------------------

        matching_h1_obfvg: list[
            str
        ] = []

        for h1_pz in h1_paired_zones:

            if (
                h1_pz.direction
                != pz.direction
            ):
                continue

            h1_low, h1_high = (
                _paired_zone_bounds(
                    h1_pz
                )
            )

            if _zones_overlap(
                h1_low,
                h1_high,
                pz_low,
                pz_high,
            ):
                matching_h1_obfvg.append(
                    (
                        "1H "
                        + (
                            "BULL"
                            if pz.direction
                            == 1
                            else "BEAR"
                        )
                        + " OB+FVG "
                        f"({h1_low:.5f}-"
                        f"{h1_high:.5f})"
                    )
                )

        # --------------------------------------------------------
        # Check 15M zone against 4H OB/FVG as additional context.
        # --------------------------------------------------------

        matching_h4_obfvg: list[
            str
        ] = []

        for h4_pz in h4_paired_zones:

            if (
                h4_pz.direction
                != pz.direction
            ):
                continue

            h4_low, h4_high = (
                _paired_zone_bounds(
                    h4_pz
                )
            )

            if _zones_overlap(
                h4_low,
                h4_high,
                pz_low,
                pz_high,
            ):
                matching_h4_obfvg.append(
                    (
                        "4H "
                        + (
                            "BULL"
                            if pz.direction
                            == 1
                            else "BEAR"
                        )
                        + " OB+FVG "
                        f"({h4_low:.5f}-"
                        f"{h4_high:.5f})"
                    )
                )

        # --------------------------------------------------------
        # The 15M zone must have meaningful HTF location.
        # --------------------------------------------------------

        htf_location_exists = bool(
            matching_htf_sr
            or matching_h1_obfvg
            or matching_h4_obfvg
        )

        if not htf_location_exists:
            continue

        # --------------------------------------------------------
        # 15M structure must agree with the direction.
        # --------------------------------------------------------

        if pz.direction == 1:

            if not bull_15m_structure:
                continue

            sweep_ok = bull_15m_sweep

        else:

            if not bear_15m_structure:
                continue

            sweep_ok = bear_15m_sweep

        # --------------------------------------------------------
        # Your intended setup favors a liquidity sweep before
        # confirmation.
        #
        # We require the relevant directional sweep here.
        # --------------------------------------------------------

        if not sweep_ok:
            continue

        primary_15m_candidates.append(
            {
                "direction": pz.direction,
                "pz": pz,
                "pz_low": pz_low,
                "pz_high": pz_high,
                "htf_sr": matching_htf_sr,
                "h1_obfvg": matching_h1_obfvg,
                "h4_obfvg": matching_h4_obfvg,
                "structure_event": (
                    "BULLISH BOS/CHoCH"
                    if pz.direction == 1
                    else "BEARISH BOS/CHoCH"
                ),
            }
        )

    # ============================================================
    # 8. 15M PRIMARY SETUP STATUS
    # ============================================================

    primary_15m_valid = bool(
        primary_15m_candidates
    )

    # ============================================================
    # 9. 5M / 1M REFINEMENT
    # ============================================================
    #
    # These timeframes can now only refine an already-valid 15M
    # setup.
    # ============================================================

    refinement_candidates: list[
        dict
    ] = []

    if primary_15m_valid:

        for primary in (
            primary_15m_candidates
        ):

            direction = (
                primary["direction"]
            )

            primary_low = (
                primary["pz_low"]
            )

            primary_high = (
                primary["pz_high"]
            )

            for ltf_name in (
                "5M",
                "1M",
            ):

                ltf_result = results[
                    ltf_name
                ]

                ltf_zones: list[
                    PairedZone
                ] = ltf_result[
                    "ob_fvg"
                ].get(
                    "active_zones",
                    [],
                )

                for pz in ltf_zones:

                    if (
                        pz.direction
                        != direction
                    ):
                        continue

                    ltf_low, ltf_high = (
                        _paired_zone_bounds(
                            pz
                        )
                    )

                    # ------------------------------------------------
                    # Lower timeframe zone must overlap the primary
                    # 15M setup.
                    # ------------------------------------------------

                    if not _zones_overlap(
                        primary_low,
                        primary_high,
                        ltf_low,
                        ltf_high,
                    ):
                        continue

                    refinement_candidates.append(
                        {
                            "direction": direction,
                            "tf": ltf_name,
                            "pz": pz,
                            "low": ltf_low,
                            "high": ltf_high,
                            "primary": primary,
                        }
                    )

    # ============================================================
    # 10. SELECT BEST PRIMARY SETUP
    # ============================================================

    selected_primary = (
        primary_15m_candidates[0]
        if primary_15m_candidates
        else None
    )

    # Prefer 1M → 5M refinement, but ONLY after 15M passes.
    selected_refinement = None

    if refinement_candidates:

        selected_refinement = sorted(
            refinement_candidates,
            key=lambda x: {
                "1M": 0,
                "5M": 1,
            }.get(
                x["tf"],
                2,
            ),
        )[0]

    # ============================================================
    # 11. RECENT SWEEPS FOR REPORTING
    # ============================================================

    sweeps_15m = _recent_sweeps(
        results["15M"]["ob_fvg"],
        8,
    )

    sweeps_5m = _recent_sweeps(
        results["5M"]["ob_fvg"],
        8,
    )

    sweeps_1m = _recent_sweeps(
        results["1M"]["ob_fvg"],
        10,
    )

    total_recent_sweeps = (
        sweeps_15m
        + sweeps_5m
        + sweeps_1m
    )

    # ============================================================
    # 12. 1M EXECUTION
    # ============================================================

    scalp_signal: (
        ScalpSignal | None
    ) = None

    df_1m = results["1M"][
        "df"
    ]

    sig_1m = results["1M"][
        "ob_fvg"
    ].get(
        "signals_df",
        pd.DataFrame(),
    )

    # ------------------------------------------------------------
    # No 15M setup = absolutely no trade.
    # ------------------------------------------------------------

    if (
        selected_primary is not None
        and not df_1m.empty
    ):

        direction = int(
            selected_primary[
                "direction"
            ]
        )

        primary_pz: PairedZone = (
            selected_primary["pz"]
        )

        primary_low = float(
            selected_primary["pz_low"]
        )

        primary_high = float(
            selected_primary["pz_high"]
        )

        # --------------------------------------------------------
        # Current closed 1M candle.
        # --------------------------------------------------------

        last_close_1m = float(
            df_1m[
                "Close"
            ].iloc[-1]
        )

        last_low_1m = float(
            df_1m[
                "Low"
            ].iloc[-1]
        )

        last_high_1m = float(
            df_1m[
                "High"
            ].iloc[-1]
        )

        atr_1m_series = _atr(
            df_1m,
            14,
        )

        atr_value = (
            atr_1m_series.iloc[-1]
            if not atr_1m_series.empty
            else float("nan")
        )

        if pd.isna(atr_value):
            atr_1m = abs(
                last_high_1m
                - last_low_1m
            )

        else:
            atr_1m = float(
                atr_value
            )

        if (
            not pd.isna(atr_1m)
            and atr_1m > 0
        ):

            # ----------------------------------------------------
            # Determine whether price is actually interacting with
            # the primary 15M zone.
            # ----------------------------------------------------

            primary_zone_touched = (
                last_low_1m
                <= primary_high
                and last_high_1m
                >= primary_low
            )

            # ----------------------------------------------------
            # Lower-timeframe refinement.
            # ----------------------------------------------------

            refinement_touched = False

            if selected_refinement is not None:

                refinement_low = float(
                    selected_refinement[
                        "low"
                    ]
                )

                refinement_high = float(
                    selected_refinement[
                        "high"
                    ]
                )

                refinement_touched = (
                    last_low_1m
                    <= refinement_high
                    and last_high_1m
                    >= refinement_low
                )

            # ----------------------------------------------------
            # 1M structural confirmation.
            #
            # It is OPTIONAL refinement.
            #
            # It cannot override the 15M setup.
            # ----------------------------------------------------

            if direction == 1:

                m1_structure_ok = (
                    _recent_column_any(
                        results["1M"][
                            "ob_fvg"
                        ],
                        "bull_bos",
                        5,
                    )
                    or
                    _recent_column_any(
                        results["1M"][
                            "ob_fvg"
                        ],
                        "choch_bull",
                        5,
                    )
                    or
                    _recent_column_any(
                        results["1M"][
                            "ob_fvg"
                        ],
                        "bull_choch",
                        5,
                    )
                )

            else:

                m1_structure_ok = (
                    _recent_column_any(
                        results["1M"][
                            "ob_fvg"
                        ],
                        "bear_bos",
                        5,
                    )
                    or
                    _recent_column_any(
                        results["1M"][
                            "ob_fvg"
                        ],
                        "choch_bear",
                        5,
                    )
                    or
                    _recent_column_any(
                        results["1M"][
                            "ob_fvg"
                        ],
                        "bear_choch",
                        5,
                    )
                )

            # ----------------------------------------------------
            # 5M structural confirmation.
            # ----------------------------------------------------

            if direction == 1:

                m5_structure_ok = (
                    _recent_column_any(
                        results["5M"][
                            "ob_fvg"
                        ],
                        "bull_bos",
                        5,
                    )
                    or
                    _recent_column_any(
                        results["5M"][
                            "ob_fvg"
                        ],
                        "choch_bull",
                        5,
                    )
                    or
                    _recent_column_any(
                        results["5M"][
                            "ob_fvg"
                        ],
                        "bull_choch",
                        5,
                    )
                )

            else:

                m5_structure_ok = (
                    _recent_column_any(
                        results["5M"][
                            "ob_fvg"
                        ],
                        "bear_bos",
                        5,
                    )
                    or
                    _recent_column_any(
                        results["5M"][
                            "ob_fvg"
                        ],
                        "choch_bear",
                        5,
                    )
                    or
                    _recent_column_any(
                        results["5M"][
                            "ob_fvg"
                        ],
                        "bear_choch",
                        5,
                    )
                )

            # ----------------------------------------------------
            # Precision condition.
            #
            # We don't force a 1M BOS if the 15M setup is already
            # valid and price has returned to the setup.
            #
            # But when lower-TF structure exists, confidence rises.
            # ----------------------------------------------------

            precision_ok = (
                primary_zone_touched
                or refinement_touched
                or m1_structure_ok
                or m5_structure_ok
            )

            if precision_ok:

                # ------------------------------------------------
                # Entry remains the latest CLOSED 1M price.
                # ------------------------------------------------

                entry = (
                    last_close_1m
                )

                # ------------------------------------------------
                # Stop placement.
                #
                # For LONG:
                #     below the deeper of primary zone / current
                #     1M low.
                #
                # For SHORT:
                #     above the deeper of primary zone / current
                #     1M high.
                # ------------------------------------------------

                if direction == 1:

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

                    risk_dist = max(
                        entry - sl,
                        atr_1m * 0.5,
                    )

                    sl = (
                        entry
                        - risk_dist
                    )

                    tp = (
                        entry
                        + risk_dist
                        * rr_target
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

                    risk_dist = max(
                        sl - entry,
                        atr_1m * 0.5,
                    )

                    sl = (
                        entry
                        + risk_dist
                    )

                    tp = (
                        entry
                        - risk_dist
                        * rr_target
                    )

                    side = "SHORT"

                # ------------------------------------------------
                # Confidence hierarchy.
                # ------------------------------------------------

                directional_sweep = (
                    bull_15m_sweep
                    if direction == 1
                    else bear_15m_sweep
                )

                if (
                    directional_sweep
                    and (
                        m5_structure_ok
                        or m1_structure_ok
                    )
                    and selected_refinement
                    is not None
                ):
                    confidence = (
                        "VERY HIGH "
                        "(4H+1H+15M "
                        "Sweep+Structure+"
                        "LTF Refinement)"
                    )

                elif (
                    directional_sweep
                    and selected_refinement
                    is not None
                ):
                    confidence = (
                        "HIGH "
                        "(HTF Location+"
                        "15M Sweep+"
                        "LTF Refinement)"
                    )

                elif directional_sweep:
                    confidence = (
                        "HIGH "
                        "(HTF Location+"
                        "15M Sweep+"
                        "Structure)"
                    )

                else:
                    confidence = (
                        "STANDARD "
                        "(HTF+15M Structure)"
                    )

                # ------------------------------------------------
                # HTF source summary.
                # ------------------------------------------------

                source_parts: list[
                    str
                ] = []

                source_parts.extend(
                    selected_primary[
                        "htf_sr"
                    ]
                )

                source_parts.extend(
                    selected_primary[
                        "h1_obfvg"
                    ]
                )

                source_parts.extend(
                    selected_primary[
                        "h4_obfvg"
                    ]
                )

                if not source_parts:
                    source_parts.append(
                        "HTF location"
                    )

                htf_zone_summary = (
                    " | ".join(
                        dict.fromkeys(
                            source_parts
                        )
                    )
                )

                # ------------------------------------------------
                # LTF summary.
                # ------------------------------------------------

                if (
                    selected_refinement
                    is not None
                ):

                    ref = (
                        selected_refinement
                    )

                    ref_pz: PairedZone = (
                        ref["pz"]
                    )

                    ref_low = (
                        ref["low"]
                    )

                    ref_high = (
                        ref["high"]
                    )

                    ltf_overlap_summary = (
                        f"15M primary "
                        f"OB+FVG "
                        f"({primary_low:.5f}-"
                        f"{primary_high:.5f}) "
                        f"| {ref['tf']} "
                        f"{'BULL' if direction == 1 else 'BEAR'} "
                        f"OB+FVG "
                        f"({ref_low:.5f}-"
                        f"{ref_high:.5f})"
                    )

                else:

                    ltf_overlap_summary = (
                        f"15M primary "
                        f"{'BULL' if direction == 1 else 'BEAR'} "
                        f"OB+FVG "
                        f"({primary_low:.5f}-"
                        f"{primary_high:.5f}) "
                        "| No lower-TF paired-zone refinement"
                    )

                # ------------------------------------------------
                # Sweep summary.
                # ------------------------------------------------

                directional_sweep_text = (
                    "YES"
                    if directional_sweep
                    else "NO"
                )

                sweep_summary = (
                    f"15M:{sweeps_15m} "
                    f"| 5M:{sweeps_5m} "
                    f"| 1M:{sweeps_1m} "
                    f"| Directional 15M sweep:"
                    f"{directional_sweep_text}"
                )

                scalp_signal = (
                    ScalpSignal(
                        direction=side,
                        confidence=confidence,
                        entry_price=round(
                            entry,
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
                        rr_ratio=rr_target,
                        htf_bias=_direction_label(
                            htf_dir
                        ),
                        htf_zone_summary=(
                            htf_zone_summary
                        ),
                        ltf_overlap_summary=(
                            ltf_overlap_summary
                        ),
                        sweep_summary=(
                            sweep_summary
                        ),
                    )
                )

    # ============================================================
    # 13. RETURN EVERYTHING
    # ============================================================
    #
    # Existing keys are preserved.
    # Additional diagnostic fields are added.
    # ============================================================

    return {
        "tf_trends": {
            "4H": t_4h,
            "1H": t_1h,
            "15M": t_15m,
            "5M": t_5m,
            "1M": t_1m,
        },

        "htf_bias": htf_dir,

        # Original overlap field retained.
        #
        # We expose the new primary 15M candidates and refinement
        # candidates separately below.
        "overlaps": (
            primary_15m_candidates
            + refinement_candidates
        ),

        "primary_15m_candidates": (
            primary_15m_candidates
        ),

        "primary_15m_valid": (
            primary_15m_valid
        ),

        "refinement_candidates": (
            refinement_candidates
        ),

        "recent_sweeps": {
            "15M": sweeps_15m,
            "5M": sweeps_5m,
            "1M": sweeps_1m,
        },

        "scalp_signal": (
            scalp_signal
        ),

        "per_tf": results,
    }
