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
# SHARED CONFIG
# ============================================================

from config import (
    APP_NAME,
    WORKER_WATCHLIST as WATCHLIST,
    STRATEGY_TIMEFRAMES,
    DEFAULT_BALANCE,
    DEFAULT_RISK_PCT,
    DEFAULT_RR_TARGET,
)

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
    log.exception("Authentication module import failed")
    AUTH_AVAILABLE = False


try:
    from broker_and_account import VirtualAccount
    ACCOUNT_AVAILABLE = True
except Exception:
    log.exception("Virtual account module import failed")
    ACCOUNT_AVAILABLE = False


try:
    import worker as worker_module
    WORKER_AVAILABLE = True
except Exception as exc:
    log.exception("Worker module import failed")
    worker_module = None
    WORKER_AVAILABLE = False
    st.error(f"Worker module import failed: {exc}")


try:
    from data_loader import fetch_ohlcv
    from chart_renderer import make_chart
    CHART_AVAILABLE = True
except Exception:
    log.exception("Chart module import failed")
    CHART_AVAILABLE = False


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
    """Log out the current user (password auth)."""
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.page = "Dashboard"
    st.rerun()


def _sync_google_user_to_supabase(email: str) -> str | None:
    """
    Ensure the Google-authenticated user exists in auth_users.
    Returns username if found or created. None on failure.
    """
    try:
        from auth_and_store import (
            _supabase_enabled,
            _sb_get_user,
            _sb_upsert_user,
        )

        if not _supabase_enabled():
            return email.split("@")[0].lower()

        existing = _sb_get_user(email)
        if existing:
            return str(existing.get("username", email.split("@")[0].lower()))

        clean_user = email.split("@")[0].lower()
        suffix = 1
        original = clean_user
        while _sb_get_user(clean_user):
            clean_user = f"{original}{suffix}"
            suffix += 1

        record = {
            "username": clean_user,
            "email": email.strip().lower(),
            "salt": "",
            "password_hash": "",
            "failed_attempts": 0,
            "locked_until": 0.0,
            "created_at": int(time.time()),
        }
        if _sb_upsert_user(record):
            log.info("Auto-created Google user: %s", clean_user)
            return clean_user
        return None
    except Exception:
        log.exception("Google user sync failed")
        return None


def authentication_screen() -> None:
    """Google sign-in OR password login."""

    st.title("🎯 DESTINY AI")
    st.caption("Intelligent Multi-Timeframe Trading Platform")
    st.divider()

    # ---- Google button (Streamlit native OIDC) ----
    st.subheader("Sign in with Google")

    if st.button("🔵 Continue with Google", use_container_width=True):
        st.login()

    st.caption("Recommended — uses your Google account.")
    st.divider()

    # ---- Password login (backup) ----
    st.subheader("Or sign in with password")

    if not AUTH_AVAILABLE:
        st.warning("Password login unavailable.")
        return

    with st.form("destiny_password_login"):
        ident = st.text_input("Username or Email", key="login_identity")
        pw = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Login", use_container_width=True)

    if submitted:
        if not ident.strip() or not pw:
            st.warning("Enter username/email and password.")
        else:
            try:
                res = authenticate_user(ident.strip(), pw)
                if getattr(res, "ok", False) and getattr(res, "username", None):
                    _login_session_user(str(res.username))
                    st.rerun()
                else:
                    st.error(getattr(res, "message", "Login failed."))
            except Exception:
                log.exception("Password login failed")
                st.error("Login failed. Check logs.")

    st.divider()
    st.caption("Don't have an account? Use Google sign-in above.")


# ============================================================
# AUTH GATE
# ============================================================

# Path 1: Google (Streamlit OIDC)
if st.user.is_logged_in:
    google_email = st.user.email
    if not st.session_state.get("username"):
        synced = _sync_google_user_to_supabase(google_email)
        if synced is None:
            st.error("Could not sync Google account. Try password login.")
            st.stop()
        st.session_state.username = synced
        st.session_state.authenticated = True
        st.session_state.page = "Dashboard"
    # Authenticated — continue

# Path 2: Password (existing session)
elif st.session_state.get("authenticated", False):
    pass  # Already logged in via password — continue

# Path 3: Not authenticated — show login screen
else:
    authentication_screen()
    st.stop()


# ============================================================
# SECTION 3: SIDEBAR NAVIGATION
# ============================================================

PAGES = [
    "Dashboard",
    "Market Scanner",
    "Chart",
    "Trading Signals",
    "Open Positions",
    "Trade History",
    "Backtest",
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

    if st.sidebar.button("Logout", use_container_width=True):
        if st.user.is_logged_in:
            st.logout()
        else:
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

    st.subheader("Market Watchlist")

    watchlist_data = [
        {"Instrument": name, "Symbol": symbol}
        for name, symbol in WATCHLIST
    ]

    st.dataframe(
        pd.DataFrame(watchlist_data),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

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
                worker_module._load_virtual_account_obj(username)
            )

        balance = float(getattr(acct, "balance", 0.0))
        positions = list(getattr(acct, "positions", []) or [])
        closed_trades = list(closed_log or [])

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Account Balance", f"${balance:,.2f}")

        with col2:
            st.metric("Open Positions", len(positions))

        with col3:
            st.metric("Closed Trades", len(closed_trades))

        st.caption(f"Virtual account belonging to {username}")

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
                "Enabled" if prefs.get("auto_exec_virtual", True) else "Disabled",
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
        log.exception("Could not load virtual account for %s", username)
        st.error("Unable to load your virtual account.")


# ============================================================
# SECTION 6: MARKET SCANNER
# ============================================================

def render_market_scanner() -> None:
    """Display the live market scanner."""

    st.subheader("Market Scanner")

    if not WORKER_AVAILABLE:
        st.error("Worker module is unavailable.")
        return

    st.caption(
        "Live market scanning and multi-timeframe strategy status."
    )

    state = {}

    try:
        if WORKER_STATE_PATH.exists():
            with open(WORKER_STATE_PATH, "r", encoding="utf-8") as fh:
                state = json.load(fh)
    except Exception:
        log.exception("Could not load worker state.")
        st.warning("Worker state could not be loaded.")

    status = state.get("status", "unknown")
    updated_utc = state.get("updated_utc", "Never")
    cycle_count = state.get("cycle_count", 0)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Scanner Status", str(status).upper())

    with col2:
        st.metric("Scan Cycles", int(cycle_count))

    with col3:
        st.metric("Last Update", str(updated_utc))

    st.divider()

    st.subheader("Scanned Instruments")

    scanned = state.get("scanned_assets", [])

    if scanned:
        rows = [{"Symbol": s, "Status": "Scanned"} for s in scanned]
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No instruments have been scanned yet.")

    st.divider()

    st.subheader("Scanner Messages")

    warnings = state.get("warnings", [])

    if warnings:
        for w in warnings:
            st.warning(str(w))
    else:
        st.success("No scanner warnings reported.")

    st.divider()

    if st.button("Run Scanner Now", use_container_width=True):
        try:
            state = {}

            if WORKER_STATE_PATH.exists():
                with open(WORKER_STATE_PATH, "r", encoding="utf-8") as fh:
                    state = json.load(fh)

            with st.spinner("Scanning markets..."):
                worker_module._scan_once(state)

            st.success("Market scan completed.")

        except Exception as exc:
            log.exception("Manual market scan failed")
            st.error(f"Market scan failed: {exc}")


# ============================================================
# SECTION 7: CHART
# ============================================================

def render_chart_page() -> None:
    """Display candlestick chart with S/R and OB+FVG overlays."""

    st.subheader("Chart")

    if not CHART_AVAILABLE:
        st.error(
            "Chart module unavailable. "
            "Check data_loader.py and chart_renderer.py."
        )
        return

    symbol_map = {name: sym for name, sym in WATCHLIST}

    col1, col2, col3 = st.columns(3)

    with col1:
        display_name = st.selectbox(
            "Symbol",
            list(symbol_map.keys()),
            key="chart_symbol",
        )

    with col2:
        timeframe = st.selectbox(
            "Timeframe",
            ["15M", "1H", "4H"],
            key="chart_tf",
        )

    with col3:
        show_sr = st.checkbox("S/R", value=True, key="chart_show_sr")
        show_ob = st.checkbox("OB+FVG", value=True, key="chart_show_ob")

    symbol = symbol_map[display_name]

    tf_config = {
        "15M": dict(interval="15m", period="14d", resample=False),
        "1H": dict(interval="1h", period="60d", resample=False),
        "4H": dict(interval="1h", period="120d", resample=True),
    }[timeframe]

    try:
        with st.spinner(f"Loading {display_name} {timeframe}..."):
            df = fetch_ohlcv(
                symbol,
                interval=tf_config["interval"],
                period=tf_config["period"],
                resample_4h=tf_config["resample"],
            )

        if df.empty:
            st.error(
                f"No data for {display_name} {timeframe}. "
                "Try a different symbol or timeframe."
            )
            return

        # Engines
        sr_zones = []
        paired_zones = []
        sweep_series = None

        try:
            from sr_engine import run_sr_zones_engine
            sr_result = run_sr_zones_engine(df, tf_label=timeframe)
            sr_zones = sr_result.get("active_zones", [])
        except Exception:
            log.exception("S/R engine failed")

        try:
            from ob_fvg_engine import run_ob_fvg_engine
            ob_result = run_ob_fvg_engine(
                df,
                timeframe_minutes={"15M": 15, "1H": 60, "4H": 240}[timeframe],
            )
            paired_zones = ob_result.get("active_zones", [])
            sig_df = ob_result.get("signals_df")
            if sig_df is not None and not sig_df.empty:
                if "swept_liquidity_count" in sig_df.columns:
                    sweep_series = sig_df["swept_liquidity_count"]
        except Exception:
            log.exception("OB/FVG engine failed")

        fig = make_chart(
            df=df,
            symbol=display_name,
            timeframe=timeframe,
            sr_zones=sr_zones,
            paired_zones=paired_zones,
            sweep_series=sweep_series,
            show_sr=show_sr,
            show_ob_fvg=show_ob,
            show_trade=False,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"{len(df)} candles · "
            f"last: {df['Close'].iloc[-1]:.5g} · "
            f"at {df.index[-1]}"
        )

    except Exception as exc:
        log.exception("Chart rendering failed")
        st.error(f"Chart failed: {exc}")


# ============================================================
# SECTION 8: TRADING SIGNALS
# ============================================================

def render_trading_signals() -> None:
    """Display generated trading signals."""

    st.subheader("Trading Signals")

    if not WORKER_STATE_PATH.exists():
        st.info("No trading signals available yet.")
        return

    try:
        with open(WORKER_STATE_PATH, "r", encoding="utf-8") as fh:
            state = json.load(fh)
    except Exception as exc:
        st.error(f"Could not load trading signals: {exc}")
        return

    signals = state.get("signals", [])

    if not signals:
        st.info(
            "No trading signals generated yet. "
            "The scanner is running, but no valid setup has been confirmed."
        )
        return

    for signal in reversed(signals):
        with st.container(border=True):
            st.markdown(f"### {signal.get('symbol', 'Unknown')}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Direction", signal.get("direction", "N/A"))
            col2.metric("Entry", signal.get("entry", "N/A"))
            col3.metric("Stop Loss", signal.get("sl", "N/A"))

            col1, col2, col3 = st.columns(3)
            col1.metric("Take Profit", signal.get("tp", "N/A"))
            col2.metric("Risk/Reward", signal.get("rr", "N/A"))
            col3.metric("Confidence", signal.get("confidence", "N/A"))

            st.write(
                "Reason:",
                signal.get("reason", "No explanation provided."),
            )


# ============================================================
# SECTION 9: OPEN POSITIONS
# ============================================================

def render_open_positions() -> None:
    st.subheader("Open Positions")

    if not WORKER_AVAILABLE:
        st.warning("Worker module unavailable.")
        return

    username = st.session_state.username

    try:
        with worker_module._account_lock(username):
            acct, closed_log, prefs = (
                worker_module._load_virtual_account_obj(username)
            )

        positions = list(getattr(acct, "positions", []) or [])

        if not positions:
            st.info("No open positions.")
            return

        rows = []
        for pos in positions:
            rows.append({
                "Symbol": getattr(pos, "symbol", ""),
                "Side": getattr(pos, "side", ""),
                "Entry": getattr(pos, "entry_price", 0.0),
                "SL": getattr(pos, "stop_loss", 0.0),
                "TP": getattr(pos, "take_profit", 0.0),
                "Units": getattr(pos, "units", 0.0),
                "Opened": getattr(pos, "opened_utc", ""),
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )
    except Exception:
        log.exception("Could not load open positions")
        st.error("Could not load open positions.")


# ============================================================
# SECTION 10: TRADE HISTORY
# ============================================================

def render_trade_history() -> None:
    st.subheader("Trade History")

    if not WORKER_AVAILABLE:
        st.warning("Worker module unavailable.")
        return

    username = st.session_state.username

    try:
        with worker_module._account_lock(username):
            acct, closed_log, prefs = (
                worker_module._load_virtual_account_obj(username)
            )

        trades = list(closed_log or [])

        if not trades:
            st.info("No closed trades yet.")
            return

        rows = []
        for t in trades:
            rows.append({
                "Symbol": t.get("symbol", ""),
                "Side": t.get("side", ""),
                "Entry": t.get("entry", 0.0),
                "Exit": t.get("exit", 0.0),
                "SL": t.get("sl", 0.0),
                "TP": t.get("tp", 0.0),
                "PnL": t.get("pnl", 0.0),
                "Closed": t.get("closed_utc", ""),
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )
    except Exception:
        log.exception("Could not load trade history")
        st.error("Could not load trade history.")


# ============================================================
# SECTION 11: ACCOUNT
# ============================================================

def render_account_page() -> None:
    st.subheader("Account")
    st.caption(f"Signed in as: **{st.session_state.username}**")
    st.divider()

    if st.user.is_logged_in:
        st.write("**Auth method:** Google")
        st.write(f"**Email:** {st.user.email}")
    else:
        st.write("**Auth method:** Password")

    st.divider()
    render_account_overview()


# ============================================================
# SECTION 12: SETTINGS
# ============================================================

def render_settings() -> None:
    st.subheader("Settings")

    st.markdown("**Strategy settings (session only)**")

    new_risk = st.slider(
        "Risk per trade (%)",
        min_value=0.5,
        max_value=10.0,
        value=float(st.session_state.risk_pct),
        step=0.5,
    )

    new_rr = st.slider(
        "Target Risk/Reward",
        min_value=1.0,
        max_value=10.0,
        value=float(st.session_state.rr_target),
        step=0.5,
    )

    if st.button("Save settings (session)", use_container_width=True):
        st.session_state.risk_pct = new_risk
        st.session_state.rr_target = new_rr
        st.success("Settings updated.")

    st.divider()
    st.caption(
        "Strategy constants live in config.py. "
        "Changes there affect both the worker and the UI after redeploy."
    )
# ============================================================
# SECTION 13: BACKTEST
# ============================================================

def render_backtest_page() -> None:
    st.subheader("Backtest")

    st.caption(
        "Replay the strategy on historical data for one symbol. "
        "Takes ~5–10 minutes per run on Streamlit Cloud."
    )

    try:
        from backtest import run_backtest
    except Exception as exc:
        st.error(f"Backtest module not available: {exc}")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        symbol_display = st.selectbox(
            "Symbol",
            [name for name, _ in WATCHLIST],
            key="bt_symbol",
        )

    with col2:
        period = st.selectbox(
            "Period (months)",
            [3, 6, 12],
            index=1,
            key="bt_period",
        )

    with col3:
        risk_pct = st.number_input(
            "Risk % per trade",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
            key="bt_risk",
        )

    symbol = dict(WATCHLIST)[symbol_display]

    if st.button("Run Backtest", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        def cb(i, total, msg):
            progress.progress(min(100, int(i / max(total, 1) * 100)))
            status.info(msg)

        try:
            with st.spinner("Running backtest..."):
                result = run_backtest(
                    symbol=symbol,
                    period_months=period,
                    risk_per_trade=risk_pct,
                    progress_callback=cb,
                )

            progress.progress(100)
            status.success(f"Done: {result.total_trades} trades")

            st.divider()
            st.subheader("Results")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Trades", result.total_trades)
            c2.metric("Win Rate", f"{result.win_rate:.1f}%")
            c3.metric("Total R", f"{result.total_r:+.2f}")
            c4.metric("Avg R", f"{result.avg_r:+.3f}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Wins", result.wins)
            c2.metric("Losses", result.losses)
            c3.metric("Profit Factor", f"{result.profit_factor:.2f}")
            c4.metric("Max DD (R)", f"-{result.max_drawdown_r:.2f}")

            c1, c2 = st.columns(2)
            c1.metric("Start Balance", f"${result.starting_balance:,.2f}")
            c2.metric("End Balance", f"${result.ending_balance:,.2f}")

            if result.trades:
                st.divider()
                st.subheader("Trade Log")

                rows = []
                for t in result.trades:
                    rows.append({
                        "Entry time": str(t.entry_time),
                        "Dir": t.direction,
                        "Entry": f"{t.entry_price:.5g}",
                        "SL": f"{t.sl:.5g}",
                        "TP": f"{t.tp:.5g}",
                        "Exit": f"{t.exit_price:.5g}",
                        "Reason": t.exit_reason,
                        "R": f"{t.r_multiple:+.2f}",
                        "Bars": t.bars_held,
                    })

                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as exc:
            log.exception("Backtest failed")
            st.error(f"Backtest failed: {exc}")

# ============================================================
# PAGE ROUTING
# ============================================================

if current_page == "Dashboard":
    render_dashboard()
    render_account_overview()

elif current_page == "Market Scanner":
    render_market_scanner()

elif current_page == "Chart":
    render_chart_page()

elif current_page == "Trading Signals":
    render_trading_signals()

elif current_page == "Open Positions":
    render_open_positions()

elif current_page == "Trade History":
    render_trade_history()

elif current_page == "Backtest":
    render_backtest_page()

elif current_page == "Account":
    render_account_page()

elif current_page == "Settings":
    render_settings()
