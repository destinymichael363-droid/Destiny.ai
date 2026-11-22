
"""
DESTINY AI — HISTORICAL RECORD STORAGE
Stores educational historical observations.
No brokerage connection or trade execution.
"""

import json
import os
from datetime import datetime, timezone

REGISTRY_FILE = "historical_records.json"


def load_records():
    """Load existing historical records."""
    if not os.path.exists(REGISTRY_FILE):
        return []

    try:
        with open(REGISTRY_FILE, "r", encoding="utf-8") as file:
            records = json.load(file)

        return records if isinstance(records, list) else []

    except (json.JSONDecodeError, OSError):
        return []


def save_records(records):
    """Save records to local JSON storage."""
    try:
        with open(REGISTRY_FILE, "w", encoding="utf-8") as file:
            json.dump(
                records,
                file,
                indent=2,
                default=str
            )

        return True

    except OSError:
        return False


def add_record(record):
    """Add a unique historical observation."""
    records = load_records()

    record_id = record.get("record_id")

    if not record_id:
        return False, "Record ID is missing."

    if any(
        item.get("record_id") == record_id
        for item in records
    ):
        return False, "Record already exists."

    record["saved_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    records.append(record)

    if not save_records(records):
        return False, "Unable to save record."

    return True, "Historical record saved."


def get_records():
    """Return stored historical observations."""
    return load_records()


def clear_records():
    """Clear the local historical registry."""
    return save_records([])

