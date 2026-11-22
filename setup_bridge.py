
# =========================================================
# DESTINY AI — HISTORICAL SETUP BRIDGE
# Validates, normalizes and identifies historical records.
# No broker integration or order execution.
# =========================================================

from __future__ import annotations

import hashlib
import math
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

VALID_DIRECTIONS = {"BULLISH", "BEARISH"}
VALID_STATUSES = {"ACTIVE", "INVALIDATED"}


# =========================================================
# SAFE VALUE HANDLING
# =========================================================

def is_missing(value):
    if value is None:
        return True

    try:
        if value != value:
            return True
    except Exception:
        pass

    text = str(value).strip().lower()

    return text in {"", "none", "nan", "nat", "null"}


def normalize_time(value):
    if is_missing(value):
        return None

    try:
        if hasattr(value, "to_pydatetime"):
            value = value.to_pydatetime()

        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)

            return value.astimezone(timezone.utc).isoformat()

        text = str(value).strip()

        parsed = datetime.fromisoformat(
            text.replace("Z", "+00:00")
        )

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(timezone.utc).isoformat()

    except (ValueError, TypeError, OverflowError):
        return None


def safe_float(value):
    if is_missing(value):
        return None

    try:
        number = float(value)

        if not math.isfinite(number):
            return None

        return number

    except (TypeError, ValueError, OverflowError):
        return None


# =========================================================
# UNIQUE HISTORICAL RECORD ID
# =========================================================

def create_setup_id(symbol, timeframe, setup):
    normalized_times = [
        normalize_time(setup.get("bos_time")),
        normalize_time(setup.get("ob_time")),
        normalize_time(setup.get("fvg_time")),
    ]

    raw = "|".join([
        str(symbol).upper().strip(),
        str(timeframe).upper().strip(),
        str(setup.get("direction", "")).upper().strip(),
        *[value or "" for value in normalized_times],
    ])

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:24]


# =========================================================
# HISTORICAL RECORD VALIDATION
# =========================================================

def validate_setup(setup):
    if not isinstance(setup, dict):
        return False, "Setup must be a dictionary."

    missing = [
        field
        for field in REQUIRED_FIELDS
        if field not in setup or is_missing(setup.get(field))
    ]

    if missing:
        return False, (
            "Missing required fields: "
            + ", ".join(missing)
        )

    direction = str(
        setup["direction"]
    ).strip().upper()

    status = str(
        setup["status"]
    ).strip().upper()

    if direction not in VALID_DIRECTIONS:
        return False, "Invalid direction."

    if status not in VALID_STATUSES:
        return False, "Invalid historical status."

    ob_high = safe_float(setup["ob_high"])
    ob_low = safe_float(setup["ob_low"])

    fvg_high = safe_float(setup["fvg_high"])
    fvg_low = safe_float(setup["fvg_low"])

    if None in (ob_high, ob_low, fvg_high, fvg_low):
        return False, "Invalid or non-finite price values."

    if ob_high <= ob_low:
        return False, "Invalid OB price range."

    if fvg_high <= fvg_low:
        return False, "Invalid FVG price range."

    for field in ("bos_time", "ob_time", "fvg_time"):
        if normalize_time(setup.get(field)) is None:
            return False, f"Invalid timestamp: {field}"

    return True, "Valid historical record."


# =========================================================
# RECORD PREPARATION
# =========================================================

def prepare_setup(symbol, timeframe, setup):
    valid, message = validate_setup(setup)

    if not valid:
        return None, message

    record = dict(setup)

    record["symbol"] = str(symbol).upper().strip()
    record["timeframe"] = str(timeframe).upper().strip()
    record["direction"] = str(
        record["direction"]
    ).upper().strip()
    record["status"] = str(
        record["status"]
    ).upper().strip()

    record["bos_time"] = normalize_time(
        record["bos_time"]
    )

    record["ob_time"] = normalize_time(
        record["ob_time"]
    )

    record["fvg_time"] = normalize_time(
        record["fvg_time"]
    )

    record["ob_high"] = safe_float(
        record["ob_high"]
    )

    record["ob_low"] = safe_float(
        record["ob_low"]
    )

    record["fvg_high"] = safe_float(
        record["fvg_high"]
    )

    record["fvg_low"] = safe_float(
        record["fvg_low"]
    )

    record["setup_id"] = create_setup_id(
        symbol,
        timeframe,
        record
    )

    record["recorded_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    record["record_type"] = "HISTORICAL_ANALYSIS"
    record["simulation_only"] = True

    return record, "Record prepared successfully."


# =========================================================
# REGISTER AND FILTER HISTORICAL RECORDS
# =========================================================

def register_setups(
    symbol,
    timeframe,
    setups,
    existing_ids=None
):
    if existing_ids is None:
        existing_ids = set()
    else:
        existing_ids = set(existing_ids)

    registered = []
    rejected = []
    duplicates = []

    received = 0

    for setup in setups:
        received += 1

        record, message = prepare_setup(
            symbol,
            timeframe,
            setup
        )

        if record is None:
            rejected.append({
                "setup": setup,
                "reason": message
            })
            continue

        setup_id = record["setup_id"]

        if setup_id in existing_ids:
            duplicates.append(setup_id)
            continue

        existing_ids.add(setup_id)
        registered.append(record)

    return {
        "registered": registered,
        "rejected": rejected,
        "duplicates": duplicates,
        "total_received": received,
        "total_registered": len(registered),
        "total_rejected": len(rejected),
        "total_duplicates": len(duplicates),
    }
