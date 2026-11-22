# =========================================================
# DESTINY AI — HISTORICAL PREFIX INTEGRITY AUDIT V1
# Controlled synthetic candles only.
# No live-market data or trading signals.
# =========================================================

import pandas as pd

from sr_engine import (
    analyze_sr,
    calculate_atr,
    find_confirmed_pivots,
)


def make_prefix_data():
    """Seven candles containing one controlled resistance pivot."""

    index = pd.date_range(
        "2025-01-01",
        periods=7,
        freq="1h",
    )

    data = pd.DataFrame(
        {
            "Open":  [99, 100, 102, 104, 107, 108, 106],
            "High":  [100, 102, 104, 107, 110, 108, 107],
            "Low":   [98, 100, 102, 104, 106, 105, 104],
            "Close": [99, 101, 103, 105, 108, 106, 105],
        },
        index=index,
    )

    return data


def make_future_extension(kind):
    """Create future candles that do not breach the early resistance."""

    if kind == "A":
        highs = [96, 95, 94, 93, 92, 91]
        lows =  [94, 93, 92, 91, 90, 89]

    elif kind == "B":
        highs = [96, 94, 95, 92, 94, 91]
        lows =  [94, 92, 93, 90, 92, 89]

    else:
        raise ValueError("Unknown extension type")

    index = pd.date_range(
        "2025-01-01 07:00",
        periods=len(highs),
        freq="1h",
    )

    return pd.DataFrame(
        {
            "Open": [
                (highs[i] + lows[i]) / 2
                for i in range(len(highs))
            ],
            "High": highs,
            "Low": lows,
            "Close": [
                (highs[i] + lows[i]) / 2
                for i in range(len(highs))
            ],
        },
        index=index,
    )


def build_extended_data(prefix, extension):
    return pd.concat([prefix, extension])


def check_atr_prefix(prefix, extended):
    """Future candles must not change earlier ATR values."""

    atr_prefix = calculate_atr(prefix, period=3)
    atr_extended = calculate_atr(extended, period=3)

    pd.testing.assert_series_equal(
        atr_prefix,
        atr_extended.loc[prefix.index],
        check_names=False,
        check_dtype=False,
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )


def check_pivot_prefix(prefix, extended):
    """Previously confirmed pivots must remain unchanged."""

    pivots_prefix = find_confirmed_pivots(
        prefix,
        length=2,
    )

    pivots_extended = find_confirmed_pivots(
        extended,
        length=2,
    )

    historical_extended = [
        pivot
        for pivot in pivots_extended
        if pivot["confirm_i"] < len(prefix)
    ]

    if pivots_prefix != historical_extended:
        raise AssertionError(
            "Future candles changed the previously confirmed pivot records."
        )

    future_pivots = [
        pivot
        for pivot in pivots_extended
        if pivot["confirm_i"] >= len(prefix)
    ]

    if any(
        pivot["confirm_i"] < len(prefix)
        for pivot in future_pivots
    ):
        raise AssertionError(
            "A future pivot was assigned an earlier confirmation index."
        )


def get_early_zone_snapshot(data, cutoff):
    """Return only zones confirmed at or before the prefix cutoff."""

    result = analyze_sr(
        data,
        pivot_length=2,
        atr_length=3,
        strength=3,
        invalid_mode="Wick",
        max_zones=40,
    )

    if result.empty:
        return result

    snapshot = result[
        result["Confirmed Time"] <= cutoff
    ].copy()

    snapshot = snapshot.sort_values(
        ["Confirmed Time", "Pivot Time", "Side"]
    ).reset_index(drop=True)

    return snapshot


def compare_zone_snapshots(prefix, extended):
    """Compare early zone values before and after future candles are added."""

    cutoff = prefix.index[-1]

    original = get_early_zone_snapshot(
        prefix,
        cutoff,
    )

    extended_snapshot = get_early_zone_snapshot(
        extended,
        cutoff,
    )

    if original.empty:
        raise AssertionError(
            "The controlled prefix did not produce an S/R zone."
        )

    if len(original) != len(extended_snapshot):
        raise AssertionError(
            "The number of earlier active zones changed after future candles were added."
        )

    columns = [
        "Side",
        "Zone Low",
        "Zone High",
        "Confirmed Time",
        "Pivot Time",
        "Rejections",
        "Strength",
        "Midline",
        "Strong",
    ]

    pd.testing.assert_frame_equal(
        original[columns].reset_index(drop=True),
        extended_snapshot[columns].reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )


def run_check(name, function):
    try:
        function()
        print(f"[PASS] {name}")
        return True

    except Exception as error:
        print(f"[FAIL] {name} — {error}")
        return False


def main():
    print("DESTINY AI — HISTORICAL PREFIX INTEGRITY AUDIT V1")
    print("Controlled synthetic candles only.")
    print("No live-market data or trading signals.")
    print()

    prefix = make_prefix_data()

    extension_a = build_extended_data(
        prefix,
        make_future_extension("A"),
    )

    extension_b = build_extended_data(
        prefix,
        make_future_extension("B"),
    )

    checks = []

    checks.append(
        run_check(
            "ATR prefix stability — Extension A",
            lambda: check_atr_prefix(prefix, extension_a),
        )
    )

    checks.append(
        run_check(
            "ATR prefix stability — Extension B",
            lambda: check_atr_prefix(prefix, extension_b),
        )
    )

    checks.append(
        run_check(
            "Confirmed pivot stability — Extension A",
            lambda: check_pivot_prefix(prefix, extension_a),
        )
    )

    checks.append(
        run_check(
            "Confirmed pivot stability — Extension B",
            lambda: check_pivot_prefix(prefix, extension_b),
        )
    )

    checks.append(
        run_check(
            "Historical zone stability — Extension A",
            lambda: compare_zone_snapshots(prefix, extension_a),
        )
    )

    checks.append(
        run_check(
            "Historical zone stability — Extension B",
            lambda: compare_zone_snapshots(prefix, extension_b),
        )
    )

    passed = sum(checks)
    failed = len(checks) - passed

    print()
    print("================================")
    print("PREFIX INTEGRITY AUDIT SUMMARY")
    print("================================")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {len(checks)}")
    print()

    print("INTERPRETATION")
    print(
        "These controlled tests check whether selected earlier "
        "ATR values, confirmed pivots, and active zones remain "
        "stable when future candles are appended."
    )
    print(
        "Passing does not prove that every engine behaviour "
        "or every historical dataset is free of look-ahead bias."
    )

    if failed == 0:
        print()
        print("Historical prefix integrity audit completed.")
    else:
        print()
        print("Historical prefix integrity audit requires investigation.")


if __name__ == "__main__":
    main()
