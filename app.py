from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DESTINY AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

log = logging.getLogger("destiny_ai")

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

WORKER_STATE_PATH = DATA_DIR / "worker_state.json"

# ============================================================
# APPLICATION SETTINGS
# ============================================================

APP_NAME = "DESTINY AI"

DEFAULT_BALANCE = 10000.0

DEFAULT_RISK_PCT = 2.0

DEFAULT_RR_TARGET = 5.0

WATCHLIST = [
    ("Gold (XAU)", "GC=F"),
    ("Crude Oil (WTI)", "CL=F"),
    ("EUR/USD", "EURUSD=X"),
    ("GBP/USD", "GBPUSD=X"),
    ("Bitcoin (BTC)", "BTC-USD"),
]

STRATEGY_TIMEFRAMES = [
    "4H",
    "1H",
    "15M",
    "5M",
    "1M",
]

# ============================================================
# CORE PROJECT IMPORTS
# ============================================================

try:
    from auth_and_store import (
        authenticate_user,
        register,
    )

    AUTH_AVAILABLE = True

except Exception:
    log.exception(
        "Authentication module import failed"
    )

    AUTH_AVAILABLE = False


try:
    from broker_and_account import VirtualAccount

    ACCOUNT_AVAILABLE = True

except Exception:
    log.exception(
        "Virtual account module import failed"
    )

    ACCOUNT_AVAILABLE = False


try:
    import worker as worker_module

    WORKER_AVAILABLE = True

except Exception:
    log.exception(
        "Worker module import failed"
    )

    worker_module = None
    WORKER_AVAILABLE = False


# ============================================================
# INITIAL SESSION STATE
# ============================================================

SESSION_DEFAULTS = {
    "authenticated": False,
    "username": "",
    "page": "Dashboard",
    "signals": [],
    "selected_symbol": "GC=F",
    "risk_pct": DEFAULT_RISK_PCT,
    "rr_target": DEFAULT_RR_TARGET,
    "auto_exec_virtual": True,
}

for key, value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value
