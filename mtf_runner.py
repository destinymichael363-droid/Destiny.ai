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


def _closed_only(df: pd.DataFrame | None) -> pd.DataFrame:
    """Drops the last (potentially still forming) candle to prevent lookahead bias."""
    if df is None or len(df) < 3:
        return pd.DataFrame()
    return df.iloc[:-1].copy()


def _zones_overlap(low_a: float, high_a: float, low_b: float, high_b: float) -> bool:
    return max(low_a, low_b) < min(high_a, high_b)


def _paired_zone_bounds(pz: PairedZone) -> tuple[float, float]:
    return min(pz.ob_low, pz.fvg_low), max(pz.ob_high, pz.fvg_high)


def run_closed_candle_5tf_cascade(
    frames_by_tf: dict[str, pd.DataFrame],
    rr_target: float = 5.0,
    atr_sl_buffer: float = 0.25,
) -> dict:
    """
    Runs closed-candle multi-timeframe analysis across 4H -> 1H -> 15M -> 5M -> 1M.
    Expects dict keys: '4H', '1H', '15M', '5M', '1M'.
    """
    tf_minutes_map = {"4H": 240, "1H": 60, "15M": 15, "5M": 5, "1M": 1}
    results: dict[str, dict] = {}

    for tf in ["4H", "1H", "15M", "5M", "1M"]:
        closed_df = _closed_only(frames_by_tf.get(tf))
        if closed_df.empty:
            results[tf] = {"df": closed_df, "sr": {"active_zones": []}, "ob_fvg": {"active_zones": [], "signals_df": pd.DataFrame()}}
            continue

        sr_res = run_sr_zones_engine(closed_df)
        ob_res = run_ob_fvg_engine(closed_df, timeframe_minutes=tf_minutes_map[tf])
        results[tf] = {"df": closed_df, "sr": sr_res, "ob_fvg": ob_res}

    def _get_trend(tf: str) -> int:
        sig_df = results[tf]["ob_fvg"].get("signals_df", pd.DataFrame())
        if sig_df.empty:
            return 0
        return int(sig_df["trend_direction"].iloc[-1])

    def _recent_sweeps(tf: str, lookback: int = 5) -> int:
        sig_df = results[tf]["ob_fvg"].get("signals_df", pd.DataFrame())
        if sig_df.empty:
            return 0
        return int(sig_df["swept_liquidity_count"].tail(lookback).sum())

    t_4h = _get_trend("4H")
    t_1h = _get_trend("1H")
    t_15m = _get_trend("15M")
    t_5m = _get_trend("5M")
    t_1m = _get_trend("1M")

    # Determine HTF Directional Bias (4H + 1H alignment)
    htf_dir = 0
    if (t_4h == 1 and t_1h in (1, 0)) or (t_1h == 1 and t_4h in (1, 0)):
        htf_dir = 1
    elif (t_4h == -1 and t_1h in (-1, 0)) or (t_1h == -1 and t_4h in (-1, 0)):
        htf_dir = -1

    # Gather HTF S/R zones (4H & 1H) + 1H Paired OB+FVG zones
    htf_sr_zones: list[tuple[str, SRZone]] = [
        ("4H", z) for z in results["4H"]["sr"].get("active_zones", [])
    ] + [
        ("1H", z) for z in results["1H"]["sr"].get("active_zones", [])
    ]
    h1_paired_zones: list[PairedZone] = results["1H"]["ob_fvg"].get("active_zones", [])

    # Gather LTF Paired OB+FVG zones (15M, 5M, 1M)
    ltf_candidates: list[tuple[str, PairedZone]] = []
    for ltf in ["15M", "5M", "1M"]:
        for pz in results[ltf]["ob_fvg"].get("active_zones", []):
            ltf_candidates.append((ltf, pz))

    # Check HTF S/R (or 1H OB+FVG) overlap with LTF OB+FVG in HTF direction
    overlaps: list[dict] = []
    for ltf_name, pz in ltf_candidates:
        if htf_dir != 0 and pz.direction != htf_dir:
            continue
        pz_low, pz_high = _paired_zone_bounds(pz)

        for htf_name, srz in htf_sr_zones:
            expected_sr_type = "support" if pz.direction == 1 else "resistance"
            if srz.zone_type != expected_sr_type:
                continue
            if _zones_overlap(srz.bottom, srz.top, pz_low, pz_high):
                overlaps.append(
                    {
                        "direction": pz.direction,
                        "htf_source": f"{htf_name} {srz.zone_type.upper()} S/R ({srz.bottom:.5f}-{srz.top:.5f})",
                        "ltf_source": f"{ltf_name} {'BULL' if pz.direction == 1 else 'BEAR'} OB+FVG ({pz_low:.5f}-{pz_high:.5f})",
                        "ltf_tf": ltf_name,
                        "pz": pz,
                    }
                )

        for h1_pz in h1_paired_zones:
            if h1_pz.direction != pz.direction:
                continue
            h1_low, h1_high = _paired_zone_bounds(h1_pz)
            if _zones_overlap(h1_low, h1_high, pz_low, pz_high):
                overlaps.append(
                    {
                        "direction": pz.direction,
                        "htf_source": f"1H {'BULL' if h1_pz.direction == 1 else 'BEAR'} OB+FVG ({h1_low:.5f}-{h1_high:.5f})",
                        "ltf_source": f"{ltf_name} {'BULL' if pz.direction == 1 else 'BEAR'} OB+FVG ({pz_low:.5f}-{pz_high:.5f})",
                        "ltf_tf": ltf_name,
                        "pz": pz,
                    }
                )

    # Evaluate 1M execution trigger on closed 1M candles
    df_1m = results["1M"]["df"]
    sig_1m = results["1M"]["ob_fvg"].get("signals_df", pd.DataFrame())
    scalp_signal: ScalpSignal | None = None

    sweeps_15m = _recent_sweeps("15M", 6)
    sweeps_5m = _recent_sweeps("5M", 8)
    sweeps_1m = _recent_sweeps("1M", 10)
    total_recent_sweeps = sweeps_15m + sweeps_5m + sweeps_1m

    if not df_1m.empty and overlaps:
        last_close_1m = float(df_1m["Close"].iloc[-1])
        last_low_1m = float(df_1m["Low"].iloc[-1])
        last_high_1m = float(df_1m["High"].iloc[-1])
        atr_1m_series = _atr(df_1m, 14)
        atr_1m = float(atr_1m_series.iloc[-1]) if not pd.isna(atr_1m_series.iloc[-1]) else abs(last_high_1m - last_low_1m)

        # Pick highest-priority overlap (prefer 5M or 1M LTF zone)
        best = sorted(overlaps, key=lambda x: {"1M": 0, "5M": 1, "15M": 2}.get(x["ltf_tf"], 3))[0]
        pz: PairedZone = best["pz"]
        pz_low, pz_high = _paired_zone_bounds(pz)

        in_zone = (last_low_1m <= pz_high) and (last_high_1m >= pz_low)
        m1_bos_ok = True
        if not sig_1m.empty:
            if pz.direction == 1:
                m1_bos_ok = bool(sig_1m["bull_bos"].tail(5).any() or sig_1m["new_bull_setup"].tail(5).any())
            else:
                m1_bos_ok = bool(sig_1m["bear_bos"].tail(5).any() or sig_1m["new_bear_setup"].tail(5).any())

        if in_zone or m1_bos_ok:
            if pz.direction == 1:
                sl = min(pz_low, last_low_1m) - (atr_1m * atr_sl_buffer)
                risk_dist = max(last_close_1m - sl, atr_1m * 0.5)
                sl = last_close_1m - risk_dist
                tp = last_close_1m + (risk_dist * rr_target)
                side: Literal["LONG", "SHORT"] = "LONG"
            else:
                sl = max(pz_high, last_high_1m) + (atr_1m * atr_sl_buffer)
                risk_dist = max(sl - last_close_1m, atr_1m * 0.5)
                sl = last_close_1m + risk_dist
                tp = last_close_1m - (risk_dist * rr_target)
                side = "SHORT"

            conf = "HIGH (Sweep + Overlap)" if total_recent_sweeps > 0 else "STANDARD (HTF-LTF Overlap)"
            scalp_signal = ScalpSignal(
                direction=side,
                confidence=conf,
                entry_price=round(last_close_1m, 5),
                stop_loss=round(sl, 5),
                take_profit=round(tp, 5),
                rr_ratio=rr_target,
                htf_bias={1: "BULLISH", -1: "BEARISH", 0: "NEUTRAL"}[htf_dir],
                htf_zone_summary=best["htf_source"],
                ltf_overlap_summary=best["ltf_source"],
                sweep_summary=f"15M:{sweeps_15m} | 5M:{sweeps_5m} | 1M:{sweeps_1m} liquidity sweeps",
            )

    return {
        "tf_trends": {"4H": t_4h, "1H": t_1h, "15M": t_15m, "5M": t_5m, "1M": t_1m},
        "htf_bias": htf_dir,
        "overlaps": overlaps,
        "recent_sweeps": {"15M": sweeps_15m, "5M": sweeps_5m, "1M": sweeps_1m},
        "scalp_signal": scalp_signal,
        "per_tf": results,
    }

