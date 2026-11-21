# =========================================================
# DESTINY AI - S/R ENGINE AUDIT (PHASE 2)
# Diagnostic only: synthetic candles; no market data or signals.
# =========================================================
import importlib
import sys
import numpy as np
import pandas as pd

ENGINE_MODULE = "sr_engine"


def make_synthetic_candles(rows=600, seed=7):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=rows, freq="h")
    drift = np.sin(np.arange(rows) / 22.0) * 8 + np.sin(np.arange(rows) / 61.0) * 15
    close = 1000 + np.cumsum(rng.normal(0, 2.0, rows)) + drift
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 0.6, rows)
    high = np.maximum(open_, close) + rng.uniform(0.1, 1.8, rows)
    low = np.minimum(open_, close) - rng.uniform(0.1, 1.8, rows)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                       "Volume": rng.integers(100, 10000, rows)}, index=index)


def make_pivot_fixture():
    n = 31
    close = np.full(n, 100.0)
    high = np.full(n, 102.0)
    low = np.full(n, 98.0)
    # A unique high pivot at index 15, with length=3 confirmation.
    high[15] = 120.0
    close[15] = 110.0
    # Keep the surrounding candles below the target pivot.
    open_ = close.copy()
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                       "Volume": np.ones(n)}, index=pd.date_range("2025-01-01", periods=n, freq="h"))


def report(name, passed, details=""):
    print(f"[{'PASS' if passed else 'CHECK'}] {name}" + (f" — {details}" if details else ""))



def zone_signature(df):
    if df is None or df.empty:
        return set()
    required = {"Side", "Zone Low", "Zone High", "Confirmed Time", "Rejections", "Strength", "Midline"}
    if not required.issubset(df.columns):
        return set()
    return {(str(r["Side"]), round(float(r["Zone Low"]), 6), round(float(r["Zone High"]), 6), str(r["Confirmed Time"]))
            for _, r in df.iterrows()}



def make_zone_fixture(side, break_level=False):
    # Construct a deterministic, flat synthetic series with one unique pivot.
    n = 60
    close = np.full(n, 100.0)
    open_ = close.copy()
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    pivot = 25
    if side == "Support":
        low[pivot] = 80.0
        close[pivot] = 90.0
        open_[pivot] = 95.0
        if break_level:
            close[pivot + 12:] = 70.0
            open_[pivot + 12:] = 75.0
            high[pivot + 12:] = 76.0
            low[pivot + 12:] = 69.0
    else:
        high[pivot] = 120.0
        close[pivot] = 110.0
        open_[pivot] = 105.0
        if break_level:
            close[pivot + 12:] = 130.0
            open_[pivot + 12:] = 125.0
            high[pivot + 12:] = 131.0
            low[pivot + 12:] = 124.0
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                        "Volume": np.ones(n)}, index=pd.date_range("2025-02-01", periods=n, freq="h")), pivot

def count_overlapping_zone_pairs(zones):
    if zones is None or zones.empty or len(zones) < 2:
        return 0
    overlaps = 0
    records = zones.to_dict("records")
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if records[i]["Side"] != records[j]["Side"]:
                continue
            if max(records[i]["Zone Low"], records[j]["Zone Low"]) <= min(records[i]["Zone High"], records[j]["Zone High"]):
                overlaps += 1
    return overlaps


def main():
    print("\nDESTINY AI — S/R HISTORICAL INTEGRITY AUDIT")
    print("Synthetic diagnostic candles only; no market data or trading signals.\n")
    try:
        engine = importlib.import_module(ENGINE_MODULE)
    except Exception as exc:
        print(f"[FAIL] Could not import {ENGINE_MODULE}.py: {exc}")
        sys.exit(1)
    if not all(hasattr(engine, x) for x in ("analyze_sr", "find_confirmed_pivots")):
        print("[FAIL] Required S/R functions missing.")
        sys.exit(1)

    data = make_synthetic_candles()
    pivots = engine.find_confirmed_pivots(data, length=8)
    zones = engine.analyze_sr(data)
    report("600 synthetic candles generated", len(data) == 600, str(len(data)))
    report("Pivot function returns list", isinstance(pivots, list), f"{len(pivots)} pivots")
    report("Zone output is DataFrame", isinstance(zones, pd.DataFrame))
    cols = {"Side", "Zone Low", "Zone High", "Confirmed Time", "Rejections", "Strength", "Midline"}
    report("Expected output fields exist", cols.issubset(zones.columns), ", ".join(zones.columns))
    if zones.empty:
        print("[CHECK] No zones; remaining zone checks skipped.")
    else:
        report("Zone side labels valid", zones["Side"].isin(["Support", "Resistance"]).all())
        report("Zone bounds ordered", (zones["Zone Low"] <= zones["Zone High"]).all())
        report("Strength within 0–3", zones["Strength"].between(0, 3).all())
        report("Strength equals capped rejection count", (zones["Strength"] == zones["Rejections"].clip(upper=3)).all())
        report("Midline is correct", np.allclose(zones["Midline"], (zones["Zone Low"] + zones["Zone High"]) / 2))
        report("Zone count <= 40", len(zones) <= 40, f"{len(zones)}")
        report("Zones newest-first", zones["Confirmed Time"].is_monotonic_decreasing)

    # Confirmation timing: the pivot must be absent before the right-side bars exist,
    # then appear once enough bars have been included.
    fixture = make_pivot_fixture()
    target_idx = 15
    length = 3
    n_before = target_idx + length
    before = engine.find_confirmed_pivots(fixture.iloc[:n_before], length=length)
    confirmed = engine.find_confirmed_pivots(fixture.iloc[:n_before + 1], length=length)
    target_before = any(p["index"] == target_idx and p["side"] == "Resistance" for p in before)
    target_after = any(p["index"] == target_idx and p["side"] == "Resistance" for p in confirmed)
    report("Pivot absent before confirmation candle", not target_before, f"target pivot index {target_idx}")
    report("Pivot present at confirmation", target_after, f"length={length}, pivot={target_idx}")

    # Targeted invalidation fixtures: diagnostic CHECK if the targeted level remains.
    for side in ("Support", "Resistance"):
        broken_data, pivot_i = make_zone_fixture(side, break_level=True)
        intact_data, _ = make_zone_fixture(side, break_level=False)
        broken_zones = engine.analyze_sr(broken_data, pivot_length=8)
        intact_zones = engine.analyze_sr(intact_data, pivot_length=8)
        pivot_time = str(broken_data.index[pivot_i])
        intact_has_pivot_zone = (not intact_zones.empty) and (
            (intact_zones["Side"].eq(side) & intact_zones["Confirmed Time"].astype(str).eq(pivot_time)).any()
        )
        broken_has_pivot_zone = (not broken_zones.empty) and (
            (broken_zones["Side"].eq(side) & broken_zones["Confirmed Time"].astype(str).eq(pivot_time)).any()
        )
        report(f"{side} fixture creates target pivot zone", bool(intact_has_pivot_zone), pivot_time)
        report(f"{side} broken target zone removed", bool(intact_has_pivot_zone and not broken_has_pivot_zone),
               "target zone absent after close-through invalidation" if not broken_has_pivot_zone else "target zone still present; review fixture/engine behavior")

    overlap_count = count_overlapping_zone_pairs(zones)
    report("Same-side overlapping zones counted (diagnostic)", True, f"{overlap_count} overlapping pairs; not an automatic defect")

    # Compare current active-zone sets at a 70% cut and full history. Differences
    # are expected from later invalidations/reactions and are shown as diagnostics.
    cut = int(len(data) * 0.70)
    prefix = engine.analyze_sr(data.iloc[:cut])
    differences = zone_signature(zones).symmetric_difference(zone_signature(prefix))
    report("Prefix/full comparison executed", True, f"{len(differences)} signatures differ; diagnostic only")
    print("\nInterpretation: differing zone sets alone do not prove look-ahead bias.")
    print("This audit does not yet certify historical replay as look-ahead-free.")
    print("Audit completed. Review the output above.")


if __name__ == "__main__":
    main()
