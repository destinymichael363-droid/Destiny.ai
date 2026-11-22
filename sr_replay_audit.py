
# =========================================================
# DESTINY AI — ADVANCED S/R REPLAY AUDIT V2
# Controlled synthetic tests only.
# =========================================================

import pandas as pd
from unittest.mock import patch

from sr_engine import analyze_sr, find_confirmed_pivots


# ---------------------------------------------------------
# 1. SYNTHETIC DATA
# ---------------------------------------------------------

def make_fixture():
    rows = 14

    data = pd.DataFrame({
        "Open": [106.0] * rows,
        "High": [107.0] * rows,
        "Low": [105.0] * rows,
        "Close": [106.0] * rows,
        "Volume": [500] * rows
    })

    data.index = pd.date_range(
        "2025-01-01",
        periods=rows,
        freq="h"
    )

    # Known resistance pivot.
    data.iloc[2, data.columns.get_loc("High")] = 110.0

    # Confirmation candle.
    data.iloc[4, data.columns.get_loc("High")] = 109.5
    data.iloc[4, data.columns.get_loc("Low")] = 108.5
    data.iloc[4, data.columns.get_loc("Close")] = 109.0

    # First touch.
    data.iloc[5, data.columns.get_loc("High")] = 109.5
    data.iloc[5, data.columns.get_loc("Low")] = 108.5
    data.iloc[5, data.columns.get_loc("Close")] = 109.0

    # Leave zone.
    data.iloc[6, data.columns.get_loc("High")] = 107.5
    data.iloc[6, data.columns.get_loc("Low")] = 106.5
    data.iloc[6, data.columns.get_loc("Close")] = 107.0

    # First rejection.
    data.iloc[7, data.columns.get_loc("High")] = 109.5
    data.iloc[7, data.columns.get_loc("Low")] = 108.5
    data.iloc[7, data.columns.get_loc("Close")] = 107.5

    # Remain inside: no duplicate count.
    data.iloc[8, data.columns.get_loc("High")] = 109.5
    data.iloc[8, data.columns.get_loc("Low")] = 108.5
    data.iloc[8, data.columns.get_loc("Close")] = 109.0

    # Leave again.
    data.iloc[9, data.columns.get_loc("High")] = 107.5
    data.iloc[9, data.columns.get_loc("Low")] = 106.5
    data.iloc[9, data.columns.get_loc("Close")] = 107.0

    # Second rejection.
    data.iloc[10, data.columns.get_loc("High")] = 109.5
    data.iloc[10, data.columns.get_loc("Low")] = 108.5
    data.iloc[10, data.columns.get_loc("Close")] = 107.5

    # Wick breach without close breach.
    data.iloc[11, data.columns.get_loc("High")] = 110.5
    data.iloc[11, data.columns.get_loc("Low")] = 109.0
    data.iloc[11, data.columns.get_loc("Close")] = 109.5

    # Close breach.
    data.iloc[12, data.columns.get_loc("High")] = 110.5
    data.iloc[12, data.columns.get_loc("Low")] = 109.0
    data.iloc[12, data.columns.get_loc("Close")] = 110.2

    return data


# ---------------------------------------------------------
# 2. ISOLATED ENGINE FIXTURE
# ---------------------------------------------------------

def get_zone(data, end, mode="Wick", touch_on_confirmation=False):
    visible = data.iloc[:end].copy()

    pivot = {
        "side": "Resistance",
        "pivot_i": 2,
        "confirm_i": 4,
        "price": 110.0
    }

    def fake_pivots(frame, length=8):
        if len(frame) > 4:
            return [pivot]
        return []

    def fake_atr(frame, period=14):
        return pd.Series(2.0, index=frame.index)

    if touch_on_confirmation:
        visible.iloc[4, visible.columns.get_loc("High")] = 109.5
        visible.iloc[4, visible.columns.get_loc("Low")] = 108.5
        visible.iloc[4, visible.columns.get_loc("Close")] = 107.5

    with patch("sr_engine.find_confirmed_pivots", side_effect=fake_pivots):
        with patch("sr_engine.calculate_atr", side_effect=fake_atr):
            zones = analyze_sr(
                visible,
                pivot_length=2,
                atr_length=1,
                strength=3,
                invalid_mode=mode
            )

    if zones.empty:
        return None

    matches = zones[zones["Pivot Time"] == data.index[2]]

    if matches.empty:
        return None

    return matches.iloc[0]


# ---------------------------------------------------------
# 3. TEST REPORTING
# ---------------------------------------------------------

def report(name, passed, detail):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name} — {detail}")
    return passed


# ---------------------------------------------------------
# 4. PIVOT CONFIRMATION
# ---------------------------------------------------------

def test_confirmation(data):
    pivots = find_confirmed_pivots(data, length=2)

    found = [
        p for p in pivots
        if p["pivot_i"] == 2
        and p["side"] == "Resistance"
    ]

    if not found:
        return False, "Expected resistance pivot missing"

    if found[0]["confirm_i"] != 4:
        return False, "Incorrect confirmation index"

    before = get_zone(data, 4)
    after = get_zone(data, 5)

    passed = before is None and after is not None

    return passed, (
        "Zone absent before confirmation and present afterward"
        if passed else "Unexpected zone confirmation timing"
    )


# ---------------------------------------------------------
# 5. CONFIRMATION CANDLE TEST
# ---------------------------------------------------------

def test_confirmation_candle(data):
    # The confirmation candle touches and closes beyond
    # the rejection boundary.
    #
    # If it is processed as a normal visit, the later
    # return can produce an extra rejection count.

    touched = get_zone(
        data,
        8,
        touch_on_confirmation=True
    )

    if touched is None:
        return False, "Zone missing"

    count = int(touched["Rejections"])

    return count == 1, (
        f"Expected 1 rejection after later completed visit; found {count}"
    )


# ---------------------------------------------------------
# 6. REJECTION COUNTING
# ---------------------------------------------------------

def test_rejections(data):
    zone = get_zone(data, 11)

    if zone is None:
        return False, "Zone missing"

    count = int(zone["Rejections"])

    return count == 2, (
        f"Expected 2 separate rejections; found {count}"
    )


def test_no_duplicates(data):
    zone = get_zone(data, 9)

    if zone is None:
        return False, "Zone missing"

    count = int(zone["Rejections"])

    return count == 1, (
        f"Expected 1 rejection; found {count}"
    )


# ---------------------------------------------------------
# 7. INVALIDATION TESTS
# ---------------------------------------------------------

def test_wick_invalidation(data):
    before = get_zone(data, 12, "Wick")
    after = get_zone(data, 13, "Wick")

    passed = before is not None and after is None

    return passed, (
        "Wick breach removes the zone"
        if passed else "Unexpected Wick invalidation"
    )


def test_close_invalidation(data):
    before = get_zone(data, 12, "Close")
    after = get_zone(data, 13, "Close")

    passed = before is not None and after is None

    return passed, (
        "Close breach removes the zone"
        if passed else "Unexpected Close invalidation"
    )


# ---------------------------------------------------------
# 8. HISTORICAL PREFIX CONSISTENCY
# ---------------------------------------------------------

def test_prefix_consistency(data):
    first = get_zone(data, 11)
    second = get_zone(data, 11)

    if first is None or second is None:
        return False, "Historical zone missing"

    fields = [
        "Zone Low",
        "Zone High",
        "Rejections",
        "Strength",
        "Midline"
    ]

    consistent = all(
        first[field] == second[field]
        for field in fields
    )

    return consistent, (
        "Repeated identical historical prefixes match"
        if consistent else "Historical prefix results differ"
    )


# ---------------------------------------------------------
# 9. MAIN AUDIT
# ---------------------------------------------------------

def main():

    print("DESTINY AI — ADVANCED S/R REPLAY AUDIT V2")
    print("Controlled synthetic candles only.")
    print("No live-market data or trading signals.\n")

    data = make_fixture()

    tests = [
        ("Pivot confirmation", lambda: test_confirmation(data)),
        ("Confirmation candle handling", lambda: test_confirmation_candle(data)),
        ("Separate rejection counting", lambda: test_rejections(data)),
        ("Duplicate rejection prevention", lambda: test_no_duplicates(data)),
        ("Wick invalidation", lambda: test_wick_invalidation(data)),
        ("Close invalidation", lambda: test_close_invalidation(data)),
        ("Historical prefix consistency", lambda: test_prefix_consistency(data))
    ]

    passed = 0
    failed = 0

    for name, function in tests:
        try:
            success, detail = function()
        except Exception as error:
            success = False
            detail = f"Test error: {error}"

        if report(name, success, detail):
            passed += 1
        else:
            failed += 1

    print("\n================================")
    print("AUDIT SUMMARY")
    print("================================")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    print("\nINTERPRETATION")
    print("These tests use controlled synthetic data.")
    print("Passing does not certify complete look-ahead freedom.")
    print("Historical consistency here checks repeatability,")
    print("not all possible future-data leakage.")

    print("\nAdvanced S/R replay audit completed.")


if __name__ == "__main__":
    main()
