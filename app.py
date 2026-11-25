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
# ============================================================
# SECTION 3: SIDEBAR NAVIGATION
# ============================================================

PAGES = [
    "Dashboard",
    "Market Scanner",
    "Trading Signals",
    "Open Positions",
    "Trade History",
    "Account",
    "Settings",
]


def render_sidebar() -> str:
    """Render the main navigation sidebar."""

    st.sidebar.title("🎯 DESTINY AI")
    st.sidebar.caption("Multi-Timeframe Trading System")
    st.sidebar.divider()

    current_page = st.session_state.get("page", "Dashboard")

    if current_page not in PAGES:
        current_page = "Dashboard"

    selected_page = st.sidebar.radio(
        "Navigation",
        PAGES,
        index=PAGES.index(current_page),
        key="main_navigation",
    )

    st.session_state.page = selected_page

    st.sidebar.divider()

    st.sidebar.markdown(
        f"**Logged in as:** {st.session_state.username}"
    )

    if st.sidebar.button(
        "Logout",
        use_container_width=True,
    ):
        _logout_user()

    return selected_page


# ============================================================
# SECTION 3B: PAGE HEADER
# ============================================================

current_page = render_sidebar()

st.title(f"🎯 DESTINY AI — {current_page}")

st.caption(
    "Intelligent Multi-Timeframe Trading Platform"
)

st.divider()

# ============================================================
# SECTION 4: DASHBOARD OVERVIEW
# ============================================================

def render_dashboard() -> None:
    """Display the main DESTINY AI system overview."""

    st.subheader("System Overview")

    # SYSTEM STATUS
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Worker Module",
            "Connected" if WORKER_AVAILABLE else "Unavailable",
        )

    with col2:
        st.metric(
            "Account Module",
            "Connected" if ACCOUNT_AVAILABLE else "Unavailable",
        )

    with col3:
        st.metric(
            "Authentication",
            "Active" if AUTH_AVAILABLE else "Unavailable",
        )

    st.divider()

    # TRADING CONFIGURATION
    st.subheader("Trading Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Risk Per Trade",
            f"{st.session_state.risk_pct:.2f}%",
        )

    with col2:
        st.metric(
            "Target Risk-to-Reward",
            f"1:{st.session_state.rr_target:.1f}",
        )

    with col3:
        st.metric(
            "Strategy Timeframes",
            len(STRATEGY_TIMEFRAMES),
        )

    st.divider()

    # WATCHLIST
    st.subheader("Market Watchlist")

    watchlist_data = [
        {
            "Instrument": name,
            "Symbol": symbol,
        }
        for name, symbol in WATCHLIST
    ]

    st.dataframe(
        pd.DataFrame(watchlist_data),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # TIMEFRAME CASCADE
    st.subheader("Multi-Timeframe Strategy")

    timeframe_columns = st.columns(5)

    for column, timeframe in zip(
        timeframe_columns,
        STRATEGY_TIMEFRAMES,
    ):
        with column:
            st.markdown(f"**{timeframe}**")

    st.caption(
        "4H market location → 1H confirmation → "
        "15M setup → 5M/1M execution."
    )


# ============================================================
# DASHBOARD PAGE ROUTING
# ============================================================

if current_page == "Dashboard":
    render_dashboard()

# ============================================================
# SECTION 5: VIRTUAL ACCOUNT OVERVIEW
# ============================================================

def render_account_overview() -> None:
    st.subheader("Virtual Trading Account")

    if not WORKER_AVAILABLE:
        st.warning("Worker module is unavailable.")
        return

    username = st.session_state.username

    try:
        with worker_module._account_lock(username):
            acct, closed_log, prefs = (
                worker_module._load_virtual_account_obj(
                    username
                )
            )

        balance = float(
            getattr(acct, "balance", 0.0)
        )

        positions = list(
            getattr(acct, "positions", []) or []
        )

        closed_trades = list(
            closed_log or []
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Account Balance",
                f"${balance:,.2f}",
            )

        with col2:
            st.metric(
                "Open Positions",
                len(positions),
            )

        with col3:
            st.metric(
                "Closed Trades",
                len(closed_trades),
            )

        st.caption(
            "Virtual account belonging to "
            f"{username}"
        )

        st.divider()

        st.markdown("**Virtual Execution Settings**")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Risk Per Trade",
                f"{float(prefs.get('risk_pct', 2.0)):.2f}%",
            )

        with col2:
            st.metric(
                "Automatic Execution",
                (
                    "Enabled"
                    if prefs.get("auto_exec_virtual", True)
                    else "Disabled"
                ),
            )

        st.markdown("**Open Positions**")

        if positions:
            position_rows = []

            for pos in positions:
                position_rows.append({
                    "Symbol": getattr(pos, "symbol", ""),
                    "Side": getattr(pos, "side", ""),
                    "Entry": getattr(pos, "entry_price", 0.0),
                    "Stop Loss": getattr(pos, "stop_loss", 0.0),
                    "Take Profit": getattr(pos, "take_profit", 0.0),
                    "Units": getattr(pos, "units", 0.0),
                })

            st.dataframe(
                pd.DataFrame(position_rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("There are currently no open positions.")

    except Exception:
        log.exception(
            "Could not load virtual account for %s",
            username,
        )
        st.error(
            "Unable to load your virtual account. "
            "Check the application logs."
        )
            # DEBUG: CHECK SAVED TRADING SIGNALS
    signals = state.get("signals", [])

    st.write(
        "DEBUG - Saved signals:",
        len(signals),
    )

    if signals:
        st.json(signals[:3])


# ============================================================
# ADD ACCOUNT OVERVIEW TO DASHBOARD
# ============================================================

if current_page == "Dashboard":
    render_account_overview()

# ============================================================
# SECTION 6: MARKET SCANNER
# ============================================================

def render_market_scanner() -> None:
    """Display the live market scanner."""

            st.warning(
            "Worker state could not be loaded."
        )

    # PASTE DEBUG CODE HERE
    signals = state.get("signals", [])

    st.write(
        "DEBUG - Saved signals:",
        len(signals),
    )

    if signals:
        st.json(signals[:3])

    # Your scanner status display continues here

    # --------------------------------------------------------
    # WORKER STATE
    # --------------------------------------------------------

    state = {}

    try:
        if WORKER_STATE_PATH.exists():
            with open(
                WORKER_STATE_PATH,
                "r",
                encoding="utf-8",
            ) as fh:
                state = json.load(fh)

    except Exception:
        log.exception(
            "Could not load worker state."
        )

        st.warning(
            "Worker state could not be loaded."
        )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    status = state.get(
        "status",
        "unknown",
    )

    updated_utc = state.get(
        "updated_utc",
        "Never",
    )

    cycle_count = state.get(
        "cycle_count",
        0,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Scanner Status",
            str(status).upper(),
        )

    with col2:
        st.metric(
            "Scan Cycles",
            int(cycle_count),
        )

    with col3:
        st.metric(
            "Last Update",
            str(updated_utc),
        )

    st.divider()

    # --------------------------------------------------------
    # SCANNED ASSETS
    # --------------------------------------------------------

    st.subheader("Scanned Instruments")

    scanned = state.get(
        "scanned_assets",
        [],
    )

    if scanned:
        rows = []

        for symbol in scanned:
            rows.append(
                {
                    "Symbol": symbol,
                    "Status": "Scanned",
                }
            )

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "No instruments have been scanned yet."
        )

    st.divider()

    # --------------------------------------------------------
    # WARNINGS
    # --------------------------------------------------------

    warnings = state.get(
        "warnings",
        [],
    )

    st.subheader("Scanner Messages")

    if warnings:
        for warning in warnings:
            st.warning(str(warning))
    else:
        st.success(
            "No scanner warnings reported."
        )

    st.divider()

           # --------------------------------------------------------
    # RUN SCANNER
    # --------------------------------------------------------

    if st.button(
        "Run Scanner Now",
        use_container_width=True,
    ):
        try:
            state = {}

            if WORKER_STATE_PATH.exists():
                with open(
                    WORKER_STATE_PATH,
                    "r",
                    encoding="utf-8",
                ) as fh:
                    state = json.load(fh)

            with st.spinner(
                "Scanning markets..."
            ):
                worker_module._scan_once(
                    state
                )

            st.success(
                "Market scan completed."
            )

            st.rerun()

        except Exception as exc:
            log.exception(
                "Manual market scan failed"
            )

            st.error(
                f"Market scan failed: {exc}"
    )
    # --------------------------------------------------------
    # STRATEGY ENGINE DIAGNOSTIC
    # --------------------------------------------------------

    if (
        WORKER_AVAILABLE
        and getattr(
            worker_module,
            "_evaluate_setups",
            None,
        ) is None
    ):
        st.error(
            "strategy_engine.py could not be loaded."
        )

        try:
            import strategy_engine

            st.success(
                "strategy_engine.py loaded successfully."
            )

        except Exception as exc:
            st.exception(exc)
# ============================================================
# MARKET SCANNER PAGE ROUTING
# ============================================================

if current_page == "Market Scanner":
    render_market_scanner()
        signals = state.get("signals", [])


