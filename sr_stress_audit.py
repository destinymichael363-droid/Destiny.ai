
# =========================================================
# DESTINY AI — HISTORICAL REPLAY STRESS TEST V1
# Controlled synthetic candles only.
# No live-market data or trading signals.
# =========================================================

import numpy as np
import pandas as pd

from sr_engine import analyze_sr


REQUIRED_COLUMNS = [
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

MAX_ZONES = 40


def make_market(scenario, length=240, seed=42):
    """Generate deterministic synthetic OHLC candles."""

    rng = np.random.default_rng(seed)

    if scenario == "Trending":
        changes = rng.normal(0.12, 0.45, length)
        close = 100 + np.cumsum(changes)

    elif scenario == "Ranging":
        changes = rng.normal(0, 0.35, length)
        close = 100 + np.cumsum(changes)
        close = 100 + (close - 100) * 0.25
        close += 2.5 * np.sin(np.arange(length) / 7)

    elif scenario == "High volatility":
        changes = rng.normal(0, 1.5, length)
        close = 100 + np.cumsum(changes)

    elif scenario == "Reversing":
        midpoint = length // 2
        first = rng.normal(0.18, 0.4, midpoint)
        second = rng.normal(-0.18, 0.4, length - midpoint)
        close = 100 + np.cumsum(np.concatenate([first, second]))

    else:
        raise ValueError("Unknown market scenario")

    previous = np.concatenate([[close[0]], close[:-1]])
    open_price = previous

    wick_size = (
        rng.uniform(0.05, 0.35, length)
        if scenario != "High volatility"
        else rng.uniform(0.2, 1.0, length)
    )

    high = np.maximum(open_price, close) + wick_size
    low = np.minimum(open_price, close) - wick_size

    index = pd.date_range(
        "2025-01-01",
        periods=length,
        freq="1h",
    )

    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
        },
        index=index,
    )


def run_engine(data):
    return analyze_sr(
        data,
        pivot_length=4,
        atr_length=14,
        strength=3,
        invalid_mode="Wick",
        max_zones=MAX_ZONES,
    )


def validate_output(result):
    """Check the basic structure and mathematical consistency."""

    if not isinstance(result, pd.DataFrame):
        raise AssertionError("Output is not a DataFrame")

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in result.columns
    ]

    if missing:
        raise AssertionError(
            f"Missing output columns: {missing}"
        )

    if len(result) > MAX_ZONES:
        raise AssertionError("Maximum zone count exceeded")

    if result.empty:
        return True

    if not result["Side"].isin(["Support", "Resistance"]).all():
        raise AssertionError("Invalid zone side")

    low = pd.to_numeric(result["Zone Low"], errors="coerce")
    high = pd.to_numeric(result["Zone High"], errors="coerce")

    if low.isna().any() or high.isna().any():
        raise AssertionError("Zone boundaries contain invalid values")

    if not np.isfinite(low).all() or not np.isfinite(high).all():
        raise AssertionError("Zone boundaries contain non-finite values")

    if not (low <= high).all():
        raise AssertionError("Zone boundaries are reversed")

    strength = pd.to_numeric(result["Strength"], errors="coerce")
    rejections = pd.to_numeric(result["Rejections"], errors="coerce")

    if strength.isna().any() or rejections.isna().any():
        raise AssertionError("Strength or rejection values are invalid")

    if not strength.between(0, 3).all():
        raise AssertionError("Strength outside 0–3")

    if (rejections < 0).any():
        raise AssertionError("Negative rejection count")

    expected_strength = rejections.clip(upper=3)

    if not np.array_equal(
        strength.to_numpy(),
        expected_strength.to_numpy(),
    ):
        raise AssertionError("Strength does not match capped rejections")

    expected_midline = (low + high) / 2
    actual_midline = pd.to_numeric(
        result["Midline"],
        errors="coerce",
    )

    if actual_midline.isna().any():
        raise AssertionError("Invalid midline")

    if not np.allclose(
        expected_midline.to_numpy(),
        actual_midline.to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    ):
        raise AssertionError("Midline calculation mismatch")

    confirmed = pd.to_datetime(
        result["Confirmed Time"],
        errors="coerce",
    )

    pivot = pd.to_datetime(
        result["Pivot Time"],
        errors="coerce",
    )

    if confirmed.isna().any() or pivot.isna().any():
        raise AssertionError("Invalid zone timestamps")

    if (pivot > confirmed).any():
        raise AssertionError("Pivot time occurs after confirmation")

    if not confirmed.is_monotonic_decreasing:
        raise AssertionError("Zones are not newest-first")

    return True


def check_scenario(scenario):
    data = make_market(scenario)

    result = run_engine(data)
    validate_output(result)

    repeated = run_engine(data)

    pd.testing.assert_frame_equal(
        result,
        repeated,
        check_dtype=False,
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )

    return len(result)


def check_history_lengths(scenario):
    counts = {}

    for length in [80, 160, 240]:
        data = make_market(
            scenario,
            length=length,
            seed=42,
        )

        result = run_engine(data)
        validate_output(result)
        counts[length] = len(result)

    return counts


def run_check(name, function):
    try:
        detail = function()
        print(f"[PASS] {name} — {detail}")
        return True

    except Exception as error:
        print(f"[FAIL] {name} — {error}")
        return False


def main():
    print("DESTINY AI — HISTORICAL REPLAY STRESS TEST V1")
    print("Controlled synthetic candles only.")
    print("No live-market data or trading signals.")
    print()

    scenarios = [
        "Trending",
        "Ranging",
        "High volatility",
        "Reversing",
    ]

    checks = []

    for scenario in scenarios:
        checks.append(
            run_check(
                f"{scenario} market — structure and repeatability",
                lambda s=scenario: (
                    f"{check_scenario(s)} zones validated"
                ),
            )
        )

        checks.append(
            run_check(
                f"{scenario} market — history lengths",
                lambda s=scenario: (
                    f"{check_history_lengths(s)}"
                ),
            )
        )

    passed = sum(checks)
    failed = len(checks) - passed

    print()
    print("================================")
    print("STRESS TEST SUMMARY")
    print("================================")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {len(checks)}")
    print()

    print("INTERPRETATION")
    print(
        "These tests examine selected S/R output properties "
        "under controlled synthetic market conditions."
    )
    print(
        "They do not establish real-market accuracy, "
        "profitability, or complete absence of look-ahead bias."
    )

    if failed == 0:
        print()
        print("Historical replay stress test completed.")
    else:
        print()
        print("Stress test requires investigation.")


if __name__ == "__main__":
    main()

