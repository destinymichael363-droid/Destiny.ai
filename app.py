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
        register_user,
    )

    AUTH_AVAILABLE = True

except Exception:
    log.exception(
        "Authentication module import failed"
    )

    AUTH_AVAILABLE = False

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
# ============================================================
# SECTION 2: AUTHENTICATION
# ============================================================

def _login_session_user(username: str) -> None:
    """Store the authenticated user's session."""

    st.session_state.authenticated = True
    st.session_state.username = username
    st.session_state.page = "Dashboard"


def _logout_user() -> None:
    """Log out the current user."""

    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.page = "Dashboard"

    st.rerun()


def authentication_screen() -> None:
    """Display login and registration forms."""

    st.title("🎯 DESTINY AI")

    st.caption(
        "Intelligent Multi-Timeframe Trading Platform"
    )

    st.divider()

    if not AUTH_AVAILABLE:
        st.error(
            "Authentication is unavailable. "
            "Please check auth_and_store.py."
        )
        return

    login_tab, register_tab = st.tabs(
        [
            "Login",
            "Create Account",
        ]
    )

    # --------------------------------------------------------
    # LOGIN
    # --------------------------------------------------------

    with login_tab:

        st.subheader("Welcome Back")

        with st.form(
            "destiny_login_form"
        ):

            ident = st.text_input(
                "Username or Email",
                key="login_identity",
            )

            pw = st.text_input(
                "Password",
                type="password",
                key="login_password",
            )

            login_submit = st.form_submit_button(
                "Login",
                use_container_width=True,
            )

        if login_submit:

            if not ident.strip() or not pw:
                st.warning(
                    "Enter your username/email and password."
                )

            else:
                try:
                    res = authenticate_user(
                        ident.strip(),
                        pw,
                    )

                    if (
                        getattr(res, "ok", False)
                        and getattr(res, "username", None)
                    ):

                        _login_session_user(
                            str(res.username)
                        )

                        st.success(
                            "Login successful."
                        )

                        st.rerun()

                    else:
                        st.error(
                            str(
                                getattr(
                                    res,
                                    "message",
                                    "Login failed.",
                                )
                            )
                        )

                except Exception:
                    log.exception(
                        "Login failed"
                    )

                    st.error(
                        "Login could not be completed. "
                        "Check the application logs."
                    )

    # --------------------------------------------------------
    # REGISTRATION
    # --------------------------------------------------------

    with register_tab:

        st.subheader("Create Your Account")

        with st.form(
            "destiny_register_form"
        ):

            new_username = st.text_input(
                "Username",
                key="register_username",
            )

            new_email = st.text_input(
                "Email Address",
                key="register_email",
            )

            new_password = st.text_input(
                "Password",
                type="password",
                key="register_password",
            )

            confirm_password = st.text_input(
                "Confirm Password",
                type="password",
                key="register_confirm_password",
            )

            register_submit = st.form_submit_button(
                "Create Account",
                use_container_width=True,
            )

        if register_submit:

            if not all(
                [
                    new_username.strip(),
                    new_email.strip(),
                    new_password,
                    confirm_password,
                ]
            ):

                st.warning(
                    "Please complete all registration fields."
                )

            elif new_password != confirm_password:

                st.error(
                    "Passwords do not match."
                )

            else:

                try:
                    res = register_user(
                        new_username.strip(),
                        new_email.strip(),
                        new_password,
                    )

                    if (
                        getattr(res, "ok", False)
                        and getattr(res, "username", None)
                    ):

                        _login_session_user(
                            str(res.username)
                        )

                        st.success(
                            "Account created successfully."
                        )

                        st.rerun()

                    else:

                        st.error(
                            str(
                                getattr(
                                    res,
                                    "message",
                                    "Registration failed.",
                                )
                            )
                        )

                except Exception:
                    log.exception(
                        "Registration failed"
                    )

                    st.error(
                        "Account registration could not be completed."
                    )


# ============================================================
# AUTHENTICATION GATE
# ============================================================

if not st.session_state.get(
    "authenticated",
    False,
):

    authentication_screen()

    st.stop()
