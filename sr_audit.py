"""Destiny AI S/R diagnostics. Educational historical-data checks only.
Place this file beside sr_engine.py and run: python sr_audit.py
This script does not connect to brokers or execute transactions.
"""
from __future__ import annotations
import importlib
import sys
import numpy as np
import pandas as pd


def make_sample(rows=80):
    idx = pd.date_range("2025-01-01", periods=rows, freq="h")
    x = np.arange(rows, dtype=float)
    close = 100 + 2.0 * np.sin(x / 3.2) + 0.035 * x
    open_ = close + 0.15 * np.sin(x / 1.7)
    high = np.maximum(open_, close) + 0.35 + 0.10 * (1 + np.sin(x / 2.1))
    low = np.minimum(open_, close) - 0.35 - 0.10 * (1 + np.cos(x / 2.4))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1000 + x}, index=idx)


def main():
    try:
        engine = importlib.import_module("sr_engine")
    except Exception as exc:
        print(f"Cannot import sr_engine.py: {exc}")
        print("Place this audit file in the same folder as sr_engine.py.")
        return 2

    data = make_sample()
    checks = []
    checks.append(("Empty input returns an empty result", engine.analyze_sr(pd.DataFrame()).empty))
    checks.append(("Missing OHLC columns returns empty", engine.analyze_sr(pd.DataFrame({"Close": [1, 2, 3]})).empty))

    pivot_length = 8
    pivots = engine.find_confirmed_pivots(data, length=pivot_length)
    pivots_confirmed = all(p["index"] + pivot_length < len(data) for p in pivots)
    checks.append(("Pivot candidates have right-side candles available", pivots_confirmed))

    result = engine.analyze_sr(data, pivot_length=pivot_length)
    required = {"Side", "Zone Low", "Zone High", "Confirmed Time", "Rejections", "Strength", "Midline"}
    checks.append(("Expected output columns exist", result.empty or required.issubset(result.columns)))
    checks.append(("Strength values stay within 0..3", result.empty or result["Strength"].between(0, 3).all()))
    checks.append(("Reaction counts are non-negative", result.empty or (result["Rejections"] >= 0).all()))

    duplicate_pairs = []
    if not result.empty:
        rows = result.to_dict("records")
        for i, a in enumerate(rows):
            for j in range(i + 1, len(rows)):
                b = rows[j]
                if a["Side"] != b["Side"]:
                    continue
                ca = (float(a["Zone Low"]) + float(a["Zone High"])) / 2
                cb = (float(b["Zone Low"]) + float(b["Zone High"])) / 2
                avg = max(abs((ca + cb) / 2), 1e-12)
                if abs(ca - cb) / avg <= 0.003:
                    duplicate_pairs.append((i, j, a["Side"], ca, cb))

    # Compare snapshots to flag future-sensitive output. This is a diagnostic, not
    # an assertion: active-zone filtering can legitimately change after later bars.
    cut = 48
    prefix = engine.analyze_sr(data.iloc[:cut], pivot_length=pivot_length)
    full = result
    def sig(df):
        if df.empty:
            return set()
        return {(r["Side"], round(float(r["Zone Low"]), 6), round(float(r["Zone High"]), 6)) for _, r in df.iterrows()}
    prefix_sig, full_sig = sig(prefix), sig(full)
    future_sensitive = prefix_sig != full_sig

    print("\nDESTINY AI — S/R AUDIT REPORT")
    print("=" * 34)
    for name, passed in checks:
        print(f"[{'PASS' if passed else 'REVIEW'}] {name}")
    print(f"\nSynthetic candles checked: {len(data)}")
    print(f"Zones returned: {len(result)}")
    print(f"Near-duplicate zone pairs flagged: {len(duplicate_pairs)}")
    for i, j, side, a, b in duplicate_pairs[:10]:
        print(f"  pair {i}/{j}: {side}, centers {a:.5f} and {b:.5f}")
    print(f"Historical-prefix comparison: {'DIFFERENT (review timing/filtering)' if future_sensitive else 'same zone set'}")
    print("\nNotes: Prefix differences can result from later invalidation/reaction scans.")
    print("This report is diagnostic; it does not modify sr_engine.py or assess trade outcomes.")
    return 1 if any(not ok for _, ok in checks) else 0


if __name__ == "__main__":
    sys.exit(main())

