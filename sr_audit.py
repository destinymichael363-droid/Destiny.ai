# =========================================================
# DESTINY AI - S/R ENGINE AUDIT
# Standalone diagnostic script. Does NOT modify sr_engine.py.
# Run: python sr_audit.py
# =========================================================

import importlib
import sys
import numpy as np
import pandas as pd

ENGINE_MODULE = "sr_engine"


def make_synthetic_candles(rows=600, seed=7):
    """Generate deterministic, non-market synthetic OHLC candles."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=rows, freq="h")
    drift = np.sin(np.arange(rows) / 22.0) * 8 + np.sin(np.arange(rows) / 61.0) * 15
    steps = rng.normal(0, 2.0, rows)
    close = 1000 + np.cumsum(steps) + drift
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 0.6, rows)
    high = np.maximum(open_, close) + rng.uniform(0.1, 1.8, rows)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1.8, rows)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                       "Volume": rng.integers(100, 10000, rows)}, index=index)


def report(name, passed, details=""):
    status = "PASS" if passed else "CHECK"
    print(f"[{status}] {name}" + (f" — {details}" if details else ""))


def zone_signature(df):
    if df is None or df.empty:
        return set()
    required = {"Side", "Zone Low", "Zone High", "Confirmed Time", "Rejections", "Strength", "Midline"}
    if not required.issubset(df.columns):
        return set()
    return {
        (str(row["Side"]), round(float(row["Zone Low"]), 6),
         round(float(row["Zone High"]), 6), str(row["Confirmed Time"]))
        for _, row in df.iterrows()
    }


def main():
    print("\nDESTINY AI — S/R ENGINE AUDIT")
    print("Synthetic diagnostic candles only; no market data or trading signals.\n")
    try:
        engine = importlib.import_module(ENGINE_MODULE)
    except Exception as exc:
        print(f"[FAIL] Could not import {ENGINE_MODULE}.py: {exc}")
        print("Place sr_audit.py beside sr_engine.py and run it in the project environment.")
        sys.exit(1)

    if not hasattr(engine, "analyze_sr") or not hasattr(engine, "find_confirmed_pivots"):
        print("[FAIL] Required functions not found: analyze_sr/find_confirmed_pivots.")
        sys.exit(1)

    data = make_synthetic_candles()
    pivots = engine.find_confirmed_pivots(data, length=8)
    zones = engine.analyze_sr(data)
    report("Synthetic data generated", len(data) == 600, f"{len(data)} candles")
    report("Pivot function returns a list", isinstance(pivots, list), f"{len(pivots)} pivots")
    report("Zone output is a DataFrame", isinstance(zones, pd.DataFrame))
    required = {"Side", "Zone Low", "Zone High", "Confirmed Time", "Rejections", "Strength", "Midline"}
    report("Expected output columns present", required.issubset(set(zones.columns)),
           ", ".join(sorted(set(zones.columns))))
    if zones.empty:
        print("[CHECK] No zones returned on synthetic sample; further checks skipped.")
        return

    report("Zone sides valid", zones["Side"].isin(["Support", "Resistance"]).all())
    report("Zone bounds valid", (zones["Zone Low"] <= zones["Zone High"]).all())
    report("Strength between 0 and 3", zones["Strength"].between(0, 3).all())
    report("Strength equals capped Rejections",
           (zones["Strength"] == zones["Rejections"].clip(upper=3)).all())
    report("Midline equals zone midpoint",
           np.allclose(zones["Midline"], (zones["Zone Low"] + zones["Zone High"]) / 2))
    report("At most 40 zones returned", len(zones) <= 40, f"{len(zones)} zones")
    report("Newest confirmation times sorted first", zones["Confirmed Time"].is_monotonic_decreasing)

    cut = int(len(data) * 0.70)
    prefix_zones = engine.analyze_sr(data.iloc[:cut])
    differences = zone_signature(zones).symmetric_difference(zone_signature(prefix_zones))
    report("Prefix/full historical comparison completed", True,
           f"{len(differences)} zone signatures differ (diagnostic, not automatic failure)")
    print("\nAudit finished. CHECK items are diagnostic flags, not proof of a defect.")
    print("This script does not change files or generate entry/exit instructions.")


if __name__ == "__main__":
    main()
