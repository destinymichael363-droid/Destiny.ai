
# =========================================================
# DESTINY AI — ADVANCED S/R REPLAY AUDIT
# Synthetic diagnostic tests only.
# =========================================================

import pandas as pd

from sr_engine import analyze_sr, find_confirmed_pivots


def make_fixture(confirmation_touch=False):
    """Build a controlled resistance-zone test fixture."""

    rows = 14

    data = pd.DataFrame({
        "Open": [109.0] * rows,
        "High": [109.5] * rows,
        "Low": [108.5] * rows,
        "Close": [109.0] * rows,
        "Volume": [500] * rows
    })

    data.index = pd.date_range(
        "2025-01-01",
        periods=rows,
        freq="h"
    )

    # Unique resistance pivot at index 2.
    data.iloc[2, data.columns.get_loc("High")] = 110.0
    data.iloc[2, data.columns.get_loc("Low")] = 108.0

    # Pivot confirmation occurs at index 4.
    if confirmation_touch:
        data.iloc[4, data.columns.get_loc("High")] = 109.5
        data.iloc[4, data.columns.get_loc("Low")] = 107.5
        data.iloc[4, data.columns.get_loc("Close")] = 107.5
    else:
        data.iloc[4, data.columns.get_loc("High")] = 107.0
        data.iloc[4, data.columns.get_loc("Low")] = 106.0
        data.iloc[4, data.columns.get_loc("Close")] = 106.5

    # First visit.
    data.iloc[5, data.columns.get_loc("High")] = 109.5
    data.iloc[5, data.columns.get_loc("Low")] = 108.5
    data.iloc[5, data.columns.get_loc("Close")] = 108.2

    # Leave zone.
    data.iloc[6, data.columns.get_loc("High")] = 107.5
    data.iloc[6, data.columns.get_loc("Low")] = 106.5
    data.iloc[6, data.columns.get_loc("Close")] = 107.0

    # Second visit: first rejection.
    data.iloc[7, data.columns.get_loc("High")] = 109.5
    data.iloc[7, data.columns.get_loc("Low")] = 108.5
    data.iloc[7, data.columns.get_loc("Close")] = 107.5

    # Remain inside: must not count again.
    for i in (8, 9):
        data.iloc[i, data.columns.get_loc("High")] = 109.5
        data.iloc[i, data.columns.get_loc("Low")] = 108.5
        data.iloc[i, data.columns.get_loc("Close")] = 107.5

    # Leave again.
    data.iloc[10, data.columns.get_loc("High")] = 107.5
    data.iloc[10, data.columns.get_loc("Low")] = 106.5
    data.iloc[10, data.columns.get_loc("Close")] = 107.0

    # Third visit: second rejection.
    data.iloc[11, data.columns.get_loc("High")] = 109.5
    data.iloc[11, data.columns.get_loc("Low")] = 108.5
    data.iloc[11, data.columns.get_loc("Close")] = 107.5

    # Wick crosses top, but close remains below top.
    data.iloc[12, data.columns.get_loc("High")] = 110.5
    data.iloc[12, data.columns.get_loc("Low")] = 109.0
    data.iloc[12, data.columns.get_loc("Close")] = 109.5

    # Close crosses top.
    data.iloc[13, data.columns.get_loc("High")] = 110.5
    data.iloc[13, data.columns.get_loc("Low")] = 109.0
    data.iloc[13, data.columns.get_loc("Close")] = 110.2

    return data


def get_target_zone(data, end, mode="Wick"):
    """Return the zone created by the known pivot at index 2."""

    visible = data.iloc[:end].copy()

    zones = analyze_sr(
        visible,
        pivot_length=2,
        atr_length=1,
        strength=3,
        invalid_mode=mode
    )

    if zones.empty:
        return None

    target_time = data.index[2]

    matches = zones[zones["Pivot Time"] == target_time]

    if matches.empty:
        return None

    return matches.iloc[0]


def report(name, passed, detail):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name} — {detail}")
    return passed


def test_pivot_confirmation(data):
    pivots = find_confirmed_pivots(data, length=2)

    target = [
        p for p in pivots
        if p["pivot_i"] == 2
        and p["side"] == "Resistance"
    ]

    if not target:
        return False, "Expected resistance pivot was not found"

    if target[0]["confirm_i"] != 4:
        return False, "Unexpected confirmation candle"

    before = get_target_zone(data, 4)
    confirmed = get_target_zone(data, 5)

    if before is not None:
        return False, "Zone appeared before confirmation"

    if confirmed is None:
        return False, "Zone missing at confirmation"

    return True, "Pivot at candle 2 confirmed at candle 4"


def test_confirmation_candle_not_counted(data):
    touched = make_fixture(confirmation_touch=True)

    zone = get_target_zone(touched, 7)

    if zone is None:
        return False, "Target zone disappeared"

    count = int(zone["Rejections"])

    if count == 0:
        return True, "Confirmation candle did not create a visit"

    return False, (
        f"Found {count} rejection(s); confirmation candle "
        "appears to have affected zone state"
    )


def test_separate_rejections(data):
    zone = get_target_zone(data, 12)

    if zone is None:
        return False, "Target zone missing"

    count = int(zone["Rejections"])

    return (
        count == 2,
        f"Expected 2 separate rejections; found {count}"
    )


def test_no_duplicate_rejections(data):
    zone = get_target_zone(data, 10)

    if zone is None:
        return False, "Target zone missing"

    count = int(zone["Rejections"])

    return (
        count == 1,
        f"Expected 1 rejection during continuous visit; found {count}"
    )


def test_wick_invalidation(data):
    before = get_target_zone(data, 12, "Wick")
    after = get_target_zone(data, 13, "Wick")

    passed = before is not None and after is None

    return (
        passed,
        "Wick breach removes zone at candle 12"
        if passed else "Unexpected Wick invalidation behaviour"
    )


def test_close_invalidation(data):
    before = get_target_zone(data, 13, "Close")
    after = get_target_zone(data, 14, "Close")

    passed = before is not None and after is None

    return (
        passed,
        "Zone survives wick-only breach and disappears after close breach"
        if passed else "Unexpected Close invalidation behaviour"
    )


def main():
    print("DESTINY AI — ADVANCED S/R REPLAY AUDIT")
    print("Controlled synthetic candles only.")
    print("No live-market data or trading signals.\n")

    data = make_fixture()

    tests = [
        ("Pivot confirmation timing",
         lambda: test_pivot_confirmation(data)),

        ("Confirmation candle handling",
         lambda: test_confirmation_candle_not_counted(data)),

        ("Separate rejection counting",
         lambda: test_separate_rejections(data)),

        ("Duplicate rejection prevention",
         lambda: test_no_duplicate_rejections(data)),

        ("Wick invalidation",
         lambda: test_wick_invalidation(data)),

        ("Close invalidation",
         lambda: test_close_invalidation(data))
    ]

    passed = 0
    failed = 0

    for name, test in tests:
        try:
            success, detail = test()
        except Exception as error:
            success = False
            detail = f"Test error: {error}"

        if report(name, success, detail):
            passed += 1
        else:
            failed += 1

    print("\n==============================")
    print("AUDIT SUMMARY")
    print("==============================")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")

    print("\nINTERPRETATION")
    print("A failed test identifies behaviour requiring investigation.")
    print("Do not modify the S/R engine until the results are reviewed.")
    print("These synthetic tests do not certify all historical data.")

    print("\nAdvanced S/R replay audit completed.")


if __name__ == "__main__":
    main()
