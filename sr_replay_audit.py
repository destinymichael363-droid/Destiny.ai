
import pandas as pd
import numpy as np

from sr_engine import analyze_sr, find_confirmed_pivots


def make_candles(rows=100):
    """Create deterministic synthetic candles."""
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 0.35, rows))

    data = pd.DataFrame({
        "Open": close - rng.normal(0, 0.15, rows),
        "High": close + rng.uniform(0.1, 0.6, rows),
        "Low": close - rng.uniform(0.1, 0.6, rows),
        "Close": close,
        "Volume": rng.integers(100, 1000, rows),
    })

    data.index = pd.date_range(
        "2025-01-01",
        periods=rows,
        freq="h"
    )

    return data


def run_replay(data):
    """Recalculate zones using only candles available at each step."""
    snapshots = []

    for end in range(20, len(data) + 1):
        visible = data.iloc[:end].copy()

        try:
            zones = analyze_sr(visible)
        except Exception as error:
            return None, f"Engine error at candle {end}: {error}"

        if not isinstance(zones, pd.DataFrame):
            return None, f"Unexpected output at candle {end}"

        snapshots.append({
            "candle": end - 1,
            "time": visible.index[-1],
            "zones": zones.copy()
        })

    return snapshots, None


def check_pivot_confirmation(data, length=3):
    pivots = find_confirmed_pivots(data.copy(), length)

    if not isinstance(pivots, list):
        return False, "Pivot function did not return a list"

    for pivot in pivots:
        pivot_i = pivot.get("pivot_i")

        if not isinstance(pivot_i, (int, np.integer)):
            return False, "Pivot has no valid pivot_i"

        confirmation_i = pivot_i + length

        if confirmation_i >= len(data):
            return False, "Invalid confirmation position"

    return True, f"{len(pivots)} pivots have valid confirmation positions"


def check_zone_visibility(snapshots):
    for snapshot in snapshots:
        zones = snapshot["zones"]

        if zones.empty:
            continue

        if "Confirmed Time" not in zones.columns:
            return False, "Confirmed Time column is missing"

        for confirmed_time in zones["Confirmed Time"].dropna():
            if confirmed_time > snapshot["time"]:
                return False, (
                    f"Future zone detected at candle {snapshot['candle']}"
                )

    return True, "No zone has a future confirmation timestamp"


def check_zone_bounds(snapshots):
    for snapshot in snapshots:
        zones = snapshot["zones"]

        required = {"Zone Low", "Zone High"}

        if not required.issubset(zones.columns):
            return False, "Zone boundary columns are missing"

        if zones.empty:
            continue

        invalid = zones["Zone Low"] > zones["Zone High"]

        if invalid.any():
            return False, (
                f"Invalid zone boundaries at candle {snapshot['candle']}"
            )

    return True, "All replayed zone boundaries are correctly ordered"


def main():
    print("DESTINY AI — S/R REPLAY AUDIT")
    print("Synthetic diagnostic candles only.")
    print("No live-market data or trade signals.\n")

    data = make_candles()

    print(f"[PASS] Synthetic candles created — {len(data)}")

    pivot_ok, pivot_message = check_pivot_confirmation(data)

    print(
        f"[{'PASS' if pivot_ok else 'FAIL'}] "
        f"Pivot confirmation — {pivot_message}"
    )

    snapshots, error = run_replay(data)

    if error:
        print(f"[FAIL] Replay execution — {error}")
        return

    print(
        f"[PASS] Candle-by-candle replay completed — "
        f"{len(snapshots)} snapshots"
    )

    visibility_ok, visibility_message = check_zone_visibility(snapshots)

    print(
        f"[{'PASS' if visibility_ok else 'FAIL'}] "
        f"Zone visibility — {visibility_message}"
    )

    bounds_ok, bounds_message = check_zone_bounds(snapshots)

    print(
        f"[{'PASS' if bounds_ok else 'FAIL'}] "
        f"Zone boundaries — {bounds_message}"
    )

    print("\nINTERPRETATION")
    print("These are preliminary replay diagnostics.")
    print("They do not yet certify rejection-count timing,")
    print("invalidation timing, or complete absence of look-ahead bias.")

    print("\nS/R replay audit completed.")


if __name__ == "__main__":
    main()

