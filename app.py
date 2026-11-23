from __future__ import annotations

import json
import os
import random
import smtplib
from email.message import EmailMessage
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from auth_and_store import authenticate_user, register_user
from broker_and_account import BrokerSession, VirtualAccount
from ob_fvg_engine import run_ob_fvg_engine
from sr_engine import run_sr_zones_engine

try:
    import mtf_runner as _mtf_runner  # type: ignore
except Exception:
    _mtf_runner = None

try:
    import notifications as _notifications  # type: ignore
except Exception:
    _notifications = None


st.set_page_config(page_title="DESTINY AI — 5-TF Scalp & Account Studio", layout="wide")

WATCHLIST: dict[str, str] = {
    # Commodities
    "Gold (XAU)": "GC=F",
    "Silver (XAG)": "SI=F",
    "Crude Oil (WTI)": "CL=F",
    "Brent Oil": "BZ=F",
    "Natural Gas": "NG=F",
    # Forex Pairs
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X",
    "USD/CAD": "USDCAD=X",
    "GBP/JPY": "GBPJPY=X",
    "CAD/JPY": "CADJPY=X",
    "CAD/CHF": "CADCHF=X",
    "GBP/NZD": "GBPNZD=X",
    # Crypto
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
}

TF_CONFIG: dict[str, tuple[str, str]] = {
    "1D": ("1d", "2y"),
    "4H": ("1h", "180d"),
    "1H": ("1h", "90d"),
    "15M": ("15m", "30d"),
    "5M": ("5m", "14d"),
    "1M": ("1m", "5d"),
}

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _safe_secrets() -> dict:
    try:
        return dict(st.secrets)
    except Exception:
        return {}


def _clean_ohlcv(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV)
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    if not all(c in out.columns for c in ["Open", "High", "Low", "Close"]):
        return pd.DataFrame(columns=OHLCV)
    if "Volume" not in out.columns:
        out["Volume"] = 0.0
    out = out[OHLCV].copy()
    for col in OHLCV:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    valid = (
        (out["High"] >= out["Open"])
        & (out["High"] >= out["Close"])
        & (out["High"] >= out["Low"])
        & (out["Low"] <= out["Open"])
        & (out["Low"] <= out["Close"])
        & (out["Low"] <= out["High"])
    )
    return out.loc[valid]


def _resample_4h(hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly is None or hourly.empty:
        return pd.DataFrame(columns=OHLCV)
    try:
        res = hourly.resample("4h", label="right", closed="right").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })
        return _clean_ohlcv(res)
    except Exception:
        return pd.DataFrame(columns=OHLCV)


@st.cache_data(ttl=45)
def fetch_tf_chart(symbol: str, tf_label: str, lookback_bars: int = 250) -> pd.DataFrame:
    interval, period = TF_CONFIG.get(tf_label, ("1h", "90d"))
    try:
        raw = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False, actions=False)
        cleaned = _clean_ohlcv(raw)
        if tf_label == "4H":
            cleaned = _resample_4h(cleaned)
        return cleaned.tail(max(100, int(lookback_bars)))
    except Exception:
        return pd.DataFrame(columns=OHLCV)


def _send_confirmation_email(to_email: str, code: str) -> tuple[bool, str]:
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_pass = os.getenv("SMTP_PASS", "").strip()
    smtp_from = os.getenv("SMTP_FROM", smtp_user).strip()

    if _notifications is not None:
        for fn_name in ("send_confirmation_email", "send_email_code", "send_verification_code"):
            fn = getattr(_notifications, fn_name, None)
            if callable(fn):
                try:
                    res = fn(to_email, code)
                    if isinstance(res, tuple) and len(res) == 2:
                        return bool(res[0]), str(res[1])
                    return True, f"Confirmation code sent to **{to_email}**."
                except Exception:
                    pass

    if not smtp_host or not smtp_user or not smtp_pass:
        return False, f"SMTP not configured yet. Use confirmation code: **{code}**"

    try:
        msg = EmailMessage()
        msg["Subject"] = "DESTINY AI — Confirm Your Email Address"
        msg["From"] = smtp_from
        msg["To"] = to_email
        msg.set_content(
            f"Welcome to DESTINY AI!\n\n"
            f"Your email confirmation code is: {code}\n\n"
            f"Enter this code on the sign-up screen to finish creating your account."
        )
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True, f"Confirmation code sent to **{to_email}**."
    except Exception as exc:
        return False, f"Could not send email ({exc}). Fallback code: **{code}**"


def _login_session_user(username: str) -> None:
    st.session_state["auth_user"] = username
    st.session_state.setdefault("virtual_account", VirtualAccount())
    st.session_state.setdefault("broker_session", BrokerSession())
    st.session_state.pop("pending_signup", None)


def _render_google_auth_section() -> None:
    has_st_login = hasattr(st, "login") and hasattr(st, "user")
    secrets_data = _safe_secrets()
    auth_cfg = secrets_data.get("auth", {}) if isinstance(secrets_data, dict) else {}
    google_configured = bool(auth_cfg.get("google") or os.getenv("GOOGLE_OAUTH_ENABLED", "").lower() == "true")

    if has_st_login and getattr(st.user, "is_logged_in", False):
        g_email = str(getattr(st.user, "email", "") or "")
        g_name = str(getattr(st.user, "name", "") or g_email.split("@")[0] or "google_user")
        clean_handle = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in g_name)[:24] or "google_user"
        if g_email:
            register_user(clean_handle, g_email, os.urandom(12).hex())
        _login_session_user(clean_handle)
        st.rerun()

    st.markdown("#### Quick Sign In / Sign Up with Google")
    if has_st_login and google_configured:
        if st.button("Continue with Google", use_container_width=True):
            st.login("google")
    else:
        st.caption(
            "To enable one-click Google sign-up/login, add `[auth]` and `[auth.google]` credentials "
            "(`client_id`, `client_secret`, `server_metadata_url`, `redirect_uri`, `cookie_secret`) "
            "in `.streamlit/secrets.toml`."
        )


def render_auth_gate() -> bool:
    if st.session_state.get("auth_user"):
        return True

    st.title("DESTINY AI — Sign In or Create Account")
    st.caption("Sign in or create an account with email confirmation to access charts, virtual trading, and worker feed.")

    _render_google_auth_section()
    st.divider()

    tab_login, tab_signup = st.tabs(["Sign In", "Create Account (Confirm Email)"])

    with tab_login:
        with st.form("login_form"):
            ident = st.text_input("Username or Email")
            pw = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In")
            if submitted:
                res = authenticate_user(ident, pw)
                if res.ok and res.username:
                    _login_session_user(res.username)
                    st.rerun()
                else:
                    st.error(res.message)

    with tab_signup:
        pending = st.session_state.get("pending_signup")
        if not pending:
            with st.form("signup_step1_form"):
                u = st.text_input("Username (3–24 letters, numbers, underscores)")
                em = st.text_input("Email
