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
                em = st.text_input("Email Address")
                pw1 = st.text_input("Password (8+ characters)", type="password")
                pw2 = st.text_input("Confirm Password", type="password")
                send_code_btn = st.form_submit_button("Send Confirmation Code")
                if send_code_btn:
                    if not em or "@" not in em:
                        st.error("Please enter a valid email address.")
                    elif pw1 != pw2:
                        st.error("Passwords do not match.")
                    elif len(pw1) < 8:
                        st.error("Password must be at least 8 characters.")
                    else:
                        code = f"{random.randint(100000, 999999)}"
                        sent_ok, delivery_msg = _send_confirmation_email(em.strip(), code)
                        st.session_state["pending_signup"] = {
                            "username": u.strip(),
                            "email": em.strip(),
                            "password": pw1,
                            "code": code,
                            "delivery_msg": delivery_msg,
                            "sent_ok": sent_ok,
                        }
                        st.rerun()
        else:
            st.info(pending["delivery_msg"])
            with st.form("signup_verify_form"):
                entered_code = st.text_input("Enter 6-Digit Confirmation Code", max_chars=6)
                col_a, col_b = st.columns(2)
                confirm_btn = col_a.form_submit_button("Confirm Email & Finish Sign Up")
                cancel_btn = col_b.form_submit_button("Change Email / Restart")

                if cancel_btn:
                    st.session_state.pop("pending_signup", None)
                    st.rerun()

                if confirm_btn:
                    if entered_code.strip() != pending["code"]:
                        st.error("Invalid confirmation code. Please check and try again.")
                    else:
                        res = register_user(pending["username"], pending["email"], pending["password"])
                        if res.ok and res.username:
                            _login_session_user(res.username)
                            st.success("Email confirmed and account created!")
                            st.rerun()
                        else:
                            st.error(res.message)

    return False


def _get_val(obj: object, names: tuple[str, ...], default: object = None) -> object:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _resolve_bar_pos(df: pd.DataFrame, obj: object, attr_names: tuple[str, ...]) -> int:
    if df.empty:
        return 0
    raw_val = _get_val(obj, attr_names)
    if raw_val is None:
        return max(0, len(df) - 30)
    if isinstance(raw_val, (int, float, np.integer, np.floating)) and not isinstance(raw_val, bool):
        return max(0, min(int(raw_val), len(df) - 1))
    try:
        idx_loc = df.index.get_indexer([raw_val], method="nearest")[0]
        if idx_loc >= 0:
            return max(0, min(int(idx_loc), len(df) - 1))
    except Exception:
        pass
    return max(0, len(df) - 30)


def _ensure_min_box_height(low_val: float, high_val: float, min_height: float) -> tuple[float, float]:
    bot = min(float(low_val), float(high_val))
    top = max(float(low_val), float(high_val))
    if (top - bot) < min_height:
        mid = (top + bot) / 2.0
        half = min_height / 2.0
        return mid - half, mid + half
    return bot, top


def _filter_close_invalidated_sr(df: pd.DataFrame, zones: list) -> list:
    if df.empty or not zones:
        return []
    closes = df["Close"].to_numpy(dtype=float)
    valid_zones = []
    for sz in zones:
        bottom = _get_val(sz, ("bottom", "low", "min_price"))
        top = _get_val(sz, ("top", "high", "max_price"))
        if bottom is None or top is None:
            continue
        b_val, t_val = float(bottom), float(top)
        if b_val > t_val:
            b_val, t_val = t_val, b_val
        if bool(_get_val(sz, ("invalidated", "broken", "mitigated"), False)):
            continue
        start_pos = _resolve_bar_pos(df, sz, ("start_idx", "idx", "bar_index", "start_bar", "time", "timestamp"))
        sub_closes = closes[start_pos:]
        if np.any(sub_closes > t_val) or np.any(sub_closes < b_val):
            continue
        valid_zones.append(sz)
    return valid_zones


def _compute_minor_sr_fallback(df: pd.DataFrame, tf_label: str, min_box_h: float) -> list[dict]:
    if len(df) < 10:
        return []
    lookback = 3 if tf_label in ("15M", "5M", "1M") else 4
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    pad = max(min_box_h * 0.65, float(np.median(highs - lows)) * 0.5)

    levels: list[dict] = []
    for i in range(lookback, len(df) - lookback):
        if highs[i] == np.max(highs[i - lookback : i + lookback + 1]):
            px = float(highs[i])
            b, t = _ensure_min_box_height(px - pad, px + pad, min_box_h)
            levels.append({"zone_type": "resistance", "start_idx": i, "bottom": b, "top": t, "tf": tf_label})
        if lows[i] == np.min(lows[i - lookback : i + lookback + 1]):
            px = float(lows[i])
            b, t = _ensure_min_box_height(px - pad, px + pad, min_box_h)
            levels.append({"zone_type": "support", "start_idx": i, "bottom": b, "top": t, "tf": tf_label})
    return levels[-4:]


def _compute_ob_fvg_overlaps_fallback(df: pd.DataFrame, min_box_h: float) -> list[dict]:
    if len(df) < 5:
        return []
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    opens = df["Open"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)

    fvgs: list[tuple[int, int, float, float]] = []
    obs: list[tuple[int, int, float, float]] = []

    for i in range(1, len(df) - 1):
        if lows[i + 1] > highs[i - 
