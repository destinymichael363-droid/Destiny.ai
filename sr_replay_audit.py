
# =========================================================
# DESTINY AI — ADVANCED S/R REPLAY AUDIT V3
# Controlled synthetic tests only.
# =========================================================

import pandas as pd
from unittest.mock import patch

from sr_engine import analyze_sr, find_confirmed_pivots


# ---------------------------------------------------------
# 1. CREATE CONTROLLED SYNTHETIC CANDLES
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
    data.iloc[4, data.columns.get_loc("High")] = 107.0
    data.iloc[4, data.columns.get_loc("Low")] = 106.0
    data.iloc[4, data.columns.get_loc("Close")] = 106.5

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

    # Remain in the same visit.
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
# 2. ISOLATE THE S/R ENGINE FOR CONTROLLED TESTING
# ---------------------------------------------------------

def get_zone(
    data,
    end,
    mode="Wick",
    touch_on_confirmation=False
):

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
        return pd.Series(
            2.0,
            index=frame.index
        )

    if touch_on_confirmation and len(visible) > 4:
        visible.iloc[4, visible.columns.get_loc("High")] = 109.5
        visible.iloc[4, visible.columns.get_loc("Low")] = 108.5
        visible.iloc[4, visible.columns.get_loc("Close")] = 107.5

    with patch(
        "sr_engine.find_confirmed_pivots",
        side_effect=fake_pivots
    ):
        with patch(
            "sr_engine.calculate_atr",
            side_effect=fake_atr
        ):
            zones = analyze_sr(
                visible,
                pivot_length=2,
                atr_length=1,
                strength=3,
                invalid_mode=mode
            )

    if zones.empty:
        return None

    matches = zones[
        zones["Pivot Time"] == data.index[2]
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


# ---------------------------------------------------------
# 3. REPORTING
# ---------------------------------------------------------

def report(name, passed, detail):

    status = "PASS" if passed else "FAIL"

    print(f"[{status}] {name} — {detail}")

    return passed


# ---------------------------------------------------------
# 4. PIVOT CONFIRMATION TEST
# ---------------------------------------------------------

def test_confirmation(data):

    pivots = find_confirmed_pivots(
        data,
        length=2
    )

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

    passed = (
        before is None
        and after is not None
    )

    return passed, (
        "Zone absent before confirmation and present afterward"
        if passed else "Unexpected confirmation timing"
    )


# ---------------------------------------------------------
# 5. STRICT CONFIRMATION-CANDLE TEST
# ---------------------------------------------------------

def test_confirmation_candle(data):

    touched = make_fixture()

    # Confirmation candle touches the zone.
    touched.iloc[4, touched.columns.get_loc("High")] = 109.5
    touched.iloc[4, touched.columns.get_loc("Low")] = 108.5
    touched.iloc[4, touched.columns.get_loc("Close")] = 107.5

    # Next candle leaves the zone.
    touched.iloc[5, touched.columns.get_loc("High")] = 107.5
    touched.iloc[5, touched.columns.get_loc("Low")] = 106.5
    touched.iloc[5, touched.columns.get_loc("Close")] = 107.0

    # Following candle returns and rejects.
    touched.iloc[6, touched.columns.get_loc("High")] = 109.5
    touched.iloc[6, touched.columns.get_loc("Low")] = 108.5
    touched.iloc[6, touched.columns.get_loc("Close")] = 107.5

    zone = get_zone(
        touched,
        7,
        touch_on_confirmation=False
    )

    if zone is None:
        return False, "Target zone missing"

    count = int(zone["Rejections"])

    return count == 0, (
        f"Expected 0 rejections if confirmation candle is excluded; "
        f"found {count}"
    )


# ---------------------------------------------------------
# 6. SEPARATE REJECTION COUNTING
# ---------------------------------------------------------

def test_rejections(data):

    zone = get_zone(data, 11)

    if zone is None:
        return False, "Target zone missing"

    count = int(zone["Rejections"])

    return count == 2, (
        f"Expected 2 separate rejections; found {count}"
    )


# ---------------------------------------------------------
# 7. DUPLICATE REJECTION PREVENTION
# ---------------------------------------------------------

def test_no_duplicates(data):

    zone = get_zone(data, 9)

    if zone is None:
        return False, "Target zone missing"

    count = int(zone["Rejections"])

    return count == 1, (
        f"Expected 1 rejection; found {count}"
    )


# ---------------------------------------------------------
# 8. WICK INVALIDATION
# ---------------------------------------------------------

def test_wick_invalidation(data):

    # Candle 11 has not yet been included.
    before = get_zone(data, 11, "Wick")

    # Candle 11 is now included.
    after = get_zone(data, 12, "Wick")

    passed = (
        before is not None
        and after is None
    )

    return passed, (
        "Zone exists before breach and disappears on wick breach"
        if passed else "Unexpected Wick invalidation timing"
    )


# ---------------------------------------------------------
# 9. CLOSE INVALIDATION
# ---------------------------------------------------------

def test_close_invalidation(data):

    # Wick crosses boundary, but candle closes inside.
    before = get_zone(data, 12, "Close")

    # Close now crosses the boundary.
    after = get_zone(data, 13, "Close")

    passed = (
        before is not None
        and after is None
    )

    return passed, (
        "Zone survives wick-only breach and disappears after close breach"
        if passed else "Unexpected Close invalidation timing"
    )


# ---------------------------------------------------------
# 10. REPEATABILITY TEST
# ---------------------------------------------------------

def test_repeatability(data):

    first = get_zone(data, 11)
    second = get_zone(data, 11)

    if first is None or second is None:
        return False, "Expected historical zone missing"

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
        "Identical historical inputs produce identical results"
        if consistent else "Repeated results differ"
    )


# ---------------------------------------------------------
# 11. MAIN AUDIT
# ---------------------------------------------------------

def main():

    print("DESTINY AI — ADVANCED S/R REPLAY AUDIT V3")
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
        ("Historical repeatability", lambda: test_repeatability(data))
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
    print("Synthetic tests check selected engine behaviours.")
    print("A failure requires investigation, not automatic code changes.")
    print("Repeatability does not prove absence of look-ahead bias.")
    print("This audit does not certify all historical market data.")

    print("\nAdvanced S/R replay audit completed.")


if __name__ == "__main__":
    main()
