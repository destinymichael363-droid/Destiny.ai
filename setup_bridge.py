
# =========================================================
# DESTINY AI — SETUP BRIDGE
# Historical Analysis Record Management
# =========================================================

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

REQUIRED_FIELDS = [
    "direction",
    "bos_time",
    "ob_time",
    "ob_high",
    "ob_low",
    "fvg_time",
    "fvg_high",
    "fvg_low",
    "status",
]


def normalize_time(value):
    """Convert timestamps into a consistent ISO format."""

    if value is None:
        return None

    try:
        if hasattr(value, "isoformat"):
            return value.isoformat()

        return str(value)

    except Exception:
        return None


def create_setup_id(symbol, timeframe, setup):
    """Create a stable identifier for a historical setup."""

    raw = "|".join([
        str(symbol).upper().strip(),
        str(timeframe).upper().strip(),
        str(setup.get("direction", "")),
        normalize_time(setup.get("bos_time")) or "",
        normalize_time(setup.get("ob_time")) or "",
        normalize_time(setup.get("fvg_time")) or "",
    ])

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:20]


def validate_setup(setup):
    """Check whether the analysis record is structurally valid."""

    if not isinstance(setup, dict):
        return False, "Setup must be a dictionary."

    missing = [
        field for field in REQUIRED_FIELDS
        if field not in setup
    ]

    if missing:
        return False, f"Missing fields: {', '.join(missing)}"

    if setup["direction"] not in ("BULLISH", "BEARISH"):
        return False, "Invalid direction field."

    if setup["status"] not in ("ACTIVE", "INVALIDATED"):
        return False, "Invalid status field."

    try:
        ob_high = float(setup["ob_high"])
        ob_low = float(setup["ob_low"])
        fvg_high = float(setup["fvg_high"])
        fvg_low = float(setup["fvg_low"])

        if ob_high <= ob_low:
            return False, "Invalid OB price range."

        if fvg_high <= fvg_low:
            return False, "Invalid FVG price range."

    except (TypeError, ValueError):
        return False, "Invalid price values."

    for field in ("bos_time", "ob_time", "fvg_time"):
        if setup.get(field) is None:
            return False, f"Missing timestamp: {field}"

    return True, "Valid setup."


def prepare_setup(symbol, timeframe, setup):
    """Validate and prepare an analysis record."""

    valid, message = validate_setup(setup)

    if not valid:
        return None, message

    record = dict(setup)

    record["symbol"] = str(symbol).upper().strip()
    record["timeframe"] = str(timeframe).upper().strip()

    record["bos_time"] = normalize_time(record["bos_time"])
    record["ob_time"] = normalize_time(record["ob_time"])
    record["fvg_time"] = normalize_time(record["fvg_time"])

    record["setup_id"] = create_setup_id(
        symbol, timeframe, record
    )

    record["recorded_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    record["record_type"] = "HISTORICAL_ANALYSIS"
    record["simulation_only"] = True

    return record, "Setup prepared successfully."


def register_setups(symbol, timeframe, setups, existing_ids=None):
    """Prepare setups while preventing duplicate records."""

    if existing_ids is None:
        existing_ids = set()

    registered = []
    rejected = []
    duplicates = []

    for setup in setups:

        result, message = prepare_setup(
            symbol, timeframe, setup
        )

        if result is None:
            rejected.append({
                "setup": setup,
                "reason": message,
            })
            continue

        setup_id = result["setup_id"]

        if setup_id in existing_ids:
            duplicates.append(setup_id)
            continue

        existing_ids.add(setup_id)
        registered.append(result)

    return {
        "registered": registered,
        "rejected": rejected,
        "duplicates": duplicates,
        "total_received": len(setups),
        "total_registered": len(registered),
        "total_rejected": len(rejected),
        "total_duplicates": len(duplicates),
    }


def export_setup_records(records):
    """Convert records into JSON-safe text."""

    return json.dumps(
        records,
        indent=2,
        default=str
    )

