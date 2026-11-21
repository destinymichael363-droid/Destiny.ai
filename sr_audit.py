
# =========================================================
# DESTINY AI - S/R ENGINE AUDIT
# PHASE 2 — CORRECTED
# Synthetic diagnostic candles only.
# No market data or trading signals.
# =========================================================

import importlib
import numpy as np
import pandas as pd

ENGINE_MODULE = "sr_engine"


def make_synthetic_candles(rows=600, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=rows, freq="h")

    drift = (
        np.sin(np.arange(rows) / 22.0) * 8
        + np.sin(np.arange(rows) / 61.0) * 15
    )

    close = (
        1000
        + np.cumsum(rng.normal(0, 2.0, rows))
        + drift
    )

    open_ = np.r_[close[0], close[:-1]] + rng.normal(
        0, 0.6, rows
    )

    high = np.maximum(open_, close) + rng.uniform(
        0.1, 1.8, rows
    )

    low = np.minimum(open_, close) - rng.uniform(
        0.1, 1.8, rows
    )

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": rng.integers(100, 10000, rows)
    }, index=dates)


def make_pivot_fixture():
    n = 31

    close = np.full(n, 100.0)
    high = np.full(n, 102.0)
    low = np.full(n, 98.0)

    high[15] = 120.0
    close[15] = 110.0

    open_ = close.copy()

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": np.ones(n)
    }, index=pd.date_range(
        "2025-01-01", periods=n, freq="h"
    ))


def make_zone_fixture(side, break_level=False):
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

    data = pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": np.ones(n)
    }, index=pd.date_range(
        "2025-02-01", periods=n, freq="h"
    ))

    return data, pivot


def report(name, passed, details=""):
    status = "PASS" if passed else "CHECK"

    suffix = f" — {details}" if details else ""

    print(f"[{status}] {name}{suffix}")


def zone_signature(df):
    if df is None or df.empty:
        return set()

    required = {
        "Side", "Zone Low", "Zone High",
        "Confirmed Time", "Rejections",
        "Strength", "Midline"
    }

    if not required.issubset(df.columns):
        return set()

    return {
        (
            str(row["Side"]),
            round(float(row["Zone Low"]), 6),
            round(float(row["Zone High"]), 6),
            str(row["Confirmed Time"])
        )
        for _, row in df.iterrows()
    }


def count_overlapping_zone_pairs(zones):
    if zones is None or zones.empty or len(zones) < 2:
        return 0

    records = zones.to_dict("records")
    overlaps = 0

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            first = records[i]
            second = records[j]

            if first["Side"] != second["Side"]:
                continue

            overlap = (
                max(first["Zone Low"], second["Zone Low"])
                <= min(first["Zone High"], second["Zone High"])
            )

            if overlap:
                overlaps += 1

    return overlaps


def get_pivot_fields(pivot):
    """
    Read the pivot format returned by sr_engine.py.
    The engine uses pivot_i rather than index.
    """

    if not isinstance(pivot, dict):
        raise TypeError(
            f"Expected pivot dictionary, got {type(pivot).__name__}"
        )

    pivot_index = pivot.get("pivot_i")
    side = pivot.get("side")

    return pivot_index, side


def contains_target_pivot(pivots, target_idx, target_side):
    for pivot in pivots:
        pivot_index, side = get_pivot_fields(pivot)

        if pivot_index == target_idx and side == target_side:
            return True

    return False


def main():
    print("\nDESTINY AI — S/R HISTORICAL INTEGRITY AUDIT")
    print(
        "Synthetic diagnostic candles only; "
        "no market data or trading signals.\n"
    )

    try:
        engine = importlib.import_module(ENGINE_MODULE)
        importlib.reload(engine)

    except Exception as exc:
        print(f"[FAIL] Could not import S/R engine: {exc}")
        return

    required_functions = (
        "analyze_sr",
        "find_confirmed_pivots"
    )

    if not all(
        hasattr(engine, name)
        for name in required_functions
    ):
        print("[FAIL] Required S/R functions are missing.")
        return

    # -----------------------------------------------------
    # 1. GENERAL ENGINE TEST
    # -----------------------------------------------------

    data = make_synthetic_candles()

    try:
        pivots = engine.find_confirmed_pivots(
            data, length=8
        )

        zones = engine.analyze_sr(data)

    except Exception as exc:
        print(f"[FAIL] Engine execution error: {exc}")
        return

    report(
        "600 synthetic candles generated",
        len(data) == 600,
        str(len(data))
    )

    report(
        "Pivot function returns list",
        isinstance(pivots, list),
        f"{len(pivots)} pivots"
    )

    report(
        "Zone output is DataFrame",
        isinstance(zones, pd.DataFrame)
    )

    if not isinstance(zones, pd.DataFrame):
        print("[FAIL] Cannot continue zone checks.")
        return

    columns = {
        "Side",
        "Zone Low",
        "Zone High",
        "Confirmed Time",
        "Rejections",
        "Strength",
        "Midline"
    }

    report(
        "Expected output fields exist",
        columns.issubset(zones.columns),
        ", ".join(zones.columns)
    )

    if not columns.issubset(zones.columns):
        print("[FAIL] Required zone fields are missing.")
        return

    if zones.empty:
        print("[CHECK] No zones detected.")

    else:
        report(
            "Zone side labels valid",
            zones["Side"].isin(
                ["Support", "Resistance"]
            ).all()
        )

        report(
            "Zone bounds ordered",
            (zones["Zone Low"] <= zones["Zone High"]).all()
        )

        report(
            "Strength within 0–3",
            zones["Strength"].between(0, 3).all()
        )

        report(
            "Strength equals capped rejection count",
            (
                zones["Strength"]
                == zones["Rejections"].clip(upper=3)
            ).all()
        )

        report(
            "Midline is correct",
            np.allclose(
                zones["Midline"],
                (
                    zones["Zone Low"]
                    + zones["Zone High"]
                ) / 2
            )
        )

        report(
            "Zone count <= 40",
            len(zones) <= 40,
            str(len(zones))
        )

        report(
            "Zones newest-first",
            zones["Confirmed Time"].is_monotonic_decreasing
        )

    # -----------------------------------------------------
    # 2. PIVOT CONFIRMATION TEST
    # -----------------------------------------------------

    fixture = make_pivot_fixture()

    target_idx = 15
    pivot_length = 3

    n_before = target_idx + pivot_length

    try:
        before = engine.find_confirmed_pivots(
            fixture.iloc[:n_before],
            length=pivot_length
        )

        confirmed = engine.find_confirmed_pivots(
            fixture.iloc[:n_before + 1],
            length=pivot_length
        )

        target_before = contains_target_pivot(
            before, target_idx, "Resistance"
        )

        target_after = contains_target_pivot(
            confirmed, target_idx, "Resistance"
        )

        report(
            "Pivot absent before confirmation candle",
            not target_before,
            f"target index {target_idx}"
        )

        report(
            "Pivot present at confirmation",
            target_after,
            f"length={pivot_length}, pivot={target_idx}"
        )

    except Exception as exc:
        print(
            f"[CHECK] Pivot confirmation test could not complete: {exc}"
        )

    # -----------------------------------------------------
    # 3. SUPPORT AND RESISTANCE FIXTURES
    # -----------------------------------------------------

    for side in ("Support", "Resistance"):

        try:
            broken_data, pivot_i = make_zone_fixture(
                side, break_level=True
            )

            intact_data, _ = make_zone_fixture(
                side, break_level=False
            )

            broken_zones = engine.analyze_sr(
                broken_data, pivot_length=8
            )

            intact_zones = engine.analyze_sr(
                intact_data, pivot_length=8
            )

            pivot_time = str(
                broken_data.index[pivot_i]
            )

            intact_has_zone = (
                not intact_zones.empty
                and (
                    intact_zones["Side"].eq(side)
                    & intact_zones["Confirmed Time"]
                    .astype(str).eq(pivot_time)
                ).any()
            )

            broken_has_zone = (
                not broken_zones.empty
                and (
                    broken_zones["Side"].eq(side)
                    & broken_zones["Confirmed Time"]
                    .astype(str).eq(pivot_time)
                ).any()
            )

            report(
                f"{side} fixture creates target pivot zone",
                bool(intact_has_zone),
                pivot_time
            )

            report(
                f"{side} broken target zone removed",
                bool(intact_has_zone and not broken_has_zone),
                "Target zone absent after close-through"
                if not broken_has_zone
                else "Target zone remains; review engine behavior"
            )

        except Exception as exc:
            print(
                f"[CHECK] {side} fixture could not complete: {exc}"
            )

    # -----------------------------------------------------
    # 4. OVERLAP DIAGNOSTIC
    # -----------------------------------------------------

    overlap_count = count_overlapping_zone_pairs(zones)

    report(
        "Same-side overlapping zones counted",
        True,
        f"{overlap_count} overlapping pairs; diagnostic only"
    )

    # -----------------------------------------------------
    # 5. PREFIX/FULL HISTORY COMPARISON
    # -----------------------------------------------------

    cut = int(len(data) * 0.70)

    prefix = engine.analyze_sr(
        data.iloc[:cut]
    )

    differences = zone_signature(
        zones
    ).symmetric_difference(
        zone_signature(prefix)
    )

    report(
        "Prefix/full comparison executed",
        True,
        f"{len(differences)} signatures differ; diagnostic only"
    )

    print("\nINTERPRETATION")
    print(
        "Different zone sets alone do not prove look-ahead bias."
    )
    print(
        "This audit does not certify historical replay "
        "as look-ahead-free."
    )
    print("\nS/R audit completed.")


if __name__ == "__main__":
    main()
