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


def _zone_bounds(zone: object) -> tuple[float, float, str] | None:
    if isinstance(zone, dict):
        low_v = zone.get("bottom", zone.get("overlap_low", zone.get("ob_low", zone.get("low"))))
        high_v = zone.get("top", zone.get("overlap_high", zone.get("ob_high", zone.get("high"))))
        z_type = str(zone.get("zone_type", zone.get("kind", zone.get("direction", "")))).lower()
    else:
        low_v = getattr(zone, "bottom", getattr(zone, "low", None))
        high_v = getattr(zone, "top", getattr(zone, "high", None))
        z_type = str(getattr(zone, "zone_type", getattr(zone, "kind", ""))).lower()

    if low_v is None or high_v is None:
        return None
    b, t = float(low_v), float(high_v)
    if b > t:
        b, t = t, b
    return b, t, z_type


def evaluate_setups(
    df: pd.DataFrame,
    symbol: str = "ASSET",
    tf_label: str = "15M",
    sr_out: dict | None = None,
    ob_out: dict | None = None,
    rr_target: float = 5.0,
    max_setups: int = 3,
) -> list[TradeSetup]:
    if df is None or df.empty or len(df) < 15:
        return []

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    price = float(last_row["Close"])
    atr_proxy = float((df["High"] - df["Low"]).tail(14).mean())
    buffer = max(atr_proxy * 0.35, abs(price) * 0.0005)

    setups: list[TradeSetup] = []
    active_sr = (sr_out or {}).get("active_zones", []) or []
    overlaps = (ob_out or {}).get("overlaps", []) or []

    # Check bullish confluence (support or bullish OB+FVG near/below current price with bounce)
    for zone in active_sr[-4:]:
        bounds = _zone_bounds(zone)
        if not bounds:
            continue
        bot, top, z_type = bounds
        is_bull_zone = "sup" in z_type or "demand" in z_type or "bull" in z_type
        is_bear_zone = "res" in z_type or "supply" in z_type or "bear" in z_type

        if is_bull_zone and price >= bot * 0.998 and float(last_row["Close"]) > float(last_row["Open"]):
            sl = round(bot - buffer, 5)
            risk = max(price - sl, buffer)
            tp = round(price + (risk * float(rr_target)), 5)
            setups.append(
                TradeSetup(
                    symbol=symbol,
                    tf_label=tf_label,
                    direction="LONG",
                    entry=round(price, 5),
                    sl=sl,
                    tp=tp,
                    rr=float(rr_target),
                    confidence="MEDIUM",
                    reason=f"Bullish rejection at {tf_label} support zone [{bot:.5f}–{top:.5f}]",
                )
            )
            break

        if is_bear_zone and price <= top * 1.002 and float(last_row["Close"]) < float(last_row["Open"]):
            sl = round(top + buffer, 5)
            risk = max(sl - price, buffer)
            tp = round(price - (risk * float(rr_target)), 5)
            setups.append(
                TradeSetup(
                    symbol=symbol,
                    tf_label=tf_label,
                    direction="SHORT",
                    entry=round(price, 5),
                    sl=sl,
                    tp=tp,
                    rr=float(rr_target),
                    confidence="MEDIUM",
                    reason=f"Bearish rejection at {tf_label} resistance zone [{bot:.5f}–{top:.5f}]",
                )
            )
            break

    # Boost to HIGH confidence if an OB+FVG overlap aligns with the same direction
    for ov in overlaps[-3:]:
        bounds = _zone_bounds(ov)
        if not bounds:
            continue
        bot, top, z_type = bounds
        for s in setups:
            if s.direction == "LONG" and (bot <= s.entry <= top * 1.01 or "1" in z_type or "bull" in z_type):
                s.confidence = "HIGH"
                s.reason += " + OB/FVG overlap confluence"
            elif s.direction == "SHORT" and (bot * 0.99 <= s.entry <= top or "-1" in z_type or "bear" in z_type):
                s.confidence = "HIGH"
                s.reason += " + OB/FVG overlap confluence"

    return setups[:max_setups]
