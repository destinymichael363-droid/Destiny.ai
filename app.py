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


st.set_page_config(page_title="DESTINY AI — 5-TF Scalp & Account Studio", layout="wide")

WATCHLIST: dict[str, str] = {
    # Commodities
    "Gold (XAU)": "GC=F",
    "Silver (XAG)": "SI=F",
    "Crude Oil (WTI)": "CL=F",
    "Brent Oil": "BZ=F",
    "Natural Gas": "NG=F",
    # Major & Cross Forex Pairs
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
    "1D": ("1d", "730d"),
    "4H": ("1h", "730d"),
    "1H": ("1h", "180d"),
    "15M": ("15m", "60d"),
    "5M": ("5m", "30d"),
    "1M": ("1m", "7d"),
}

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


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
def fetch_tf_chart(symbol: str, tf_label: str, lookback_bars: int = 500) -> pd.DataFrame:
    interval, period = TF_CONFIG.get(tf_label, ("1h", "180d"))
    try:
        raw = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False, actions=False, prepost=False)
        cleaned = _clean_ohlcv(raw)
        if tf_label == "4H":
            cleaned = _resample_4h(cleaned)
        return cleaned.tail(max(120, int(lookback_bars)))
    except Exception:
        return pd.DataFrame(columns=OHLCV)


def _send_confirmation_email(to_email: str, code: str) -> tuple[bool, str]:
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_pass = os.getenv("SMTP_PASS", "").strip()
    smtp_from = os.getenv("SMTP_FROM", smtp_user).strip()

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
    auth_cfg = st.secrets.get("auth", {}) if hasattr(st, "secrets") else {}
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
        if lows[i + 1] > highs[i - 1]:
            fvgs.append((1, i, float(highs[i - 1]), float(lows[i + 1])))
        if highs[i + 1] < lows[i - 1]:
            fvgs.append((-1, i, float(highs[i + 1]), float(lows[i - 1])))
        if closes[i - 1] < opens[i - 1] and closes[i] > opens[i] and closes[i] > highs[i - 1]:
            obs.append((1, i - 1, float(lows[i - 1]), float(highs[i - 1])))
        if closes[i - 1] > opens[i - 1] and closes[i] < opens[i] and closes[i] < lows[i - 1]:
            obs.append((-1, i - 1, float(lows[i - 1]), float(highs[i - 1])))

    overlaps: list[dict] = []
    for f_dir, f_idx, f_bot, f_top in fvgs[-12:]:
        for o_dir, o_idx, o_bot, o_top in obs[-12:]:
            if f_dir == o_dir and abs(f_idx - o_idx) <= 10:
                ov_bot = max(f_bot, o_bot)
                ov_top = min(f_top, o_top)
                if ov_top > ov_bot:
                    b, t = _ensure_min_box_height(ov_bot, ov_top, min_box_h * 1.3)
                    overlaps.append({
                        "direction": f_dir,
                        "ob_start_idx": min(f_idx, o_idx),
                        "overlap_low": b,
                        "overlap_high": t,
                    })
    return overlaps[-3:]


def _compute_bos_fallback(df: pd.DataFrame) -> list[tuple[int, int, float, str]]:
    if len(df) < 10:
        return []
    swing_n = 3
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)

    last_swing_high: tuple[int, float] | None = None
    last_swing_low: tuple[int, float] | None = None
    bos_events: list[tuple[int, int, float, str]] = []

    for i in range(swing_n, len(df) - swing_n):
        if highs[i] == np.max(highs[i - swing_n : i + swing_n + 1]):
            last_swing_high = (i, float(highs[i]))
        if lows[i] == np.min(lows[i - swing_n : i + swing_n + 1]):
            last_swing_low = (i, float(lows[i]))

    for i in range(swing_n + 1, len(df)):
        if last_swing_high is not None and i > last_swing_high[0] and closes[i] > last_swing_high[1]:
            bos_events.append((last_swing_high[0], i, last_swing_high[1], "BOS ↑"))
            last_swing_high = None
        if last_swing_low is not None and i > last_swing_low[0] and closes[i] < last_swing_low[1]:
            bos_events.append((last_swing_low[0], i, last_swing_low[1], "BOS ↓"))
            last_swing_low = None

    return bos_events[-4:]


def _add_zone_box(
    fig: go.Figure,
    x0_pos: int,
    x1_pos: int,
    y_bot: float,
    y_top: float,
    fill_rgba: str,
    line_rgba: str,
    label_text: str,
    label_color: str,
    dash: str = "solid",
) -> None:
    """Draws a visible filled box + border + label using integer bar coordinates."""
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=x0_pos,
        x1=x1_pos,
        y0=y_bot,
        y1=y_top,
        fillcolor=fill_rgba,
        line=dict(color=line_rgba, width=1.5, dash=dash),
        layer="below",
    )
    fig.add_annotation(
        xref="x",
        yref="y",
        x=x1_pos,
        y=(y_bot + y_top) / 2.0,
        text=label_text,
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        bgcolor="rgba(18, 22, 30, 0.85)",
        bordercolor=line_rgba,
        borderwidth=1,
        font=dict(size=10, color=label_color),
    )


def build_chart(
    df: pd.DataFrame,
    title: str,
    tf_label: str,
    sr_zones: list,
    minor_sr_zones: list,
    paired_zones: list,
    sig_df: pd.DataFrame,
    show_minor_sr: bool = True,
    show_bos: bool = True,
    show_ob_fvg_overlap: bool = True,
    initial_visible_bars: int = 120,
) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title=f"{title} (No data)", template="plotly_dark", height=560)
        return fig

    n_bars = len(df)
    x_positions = list(range(n_bars))
    x_end_pos = n_bars - 1

    # Readable tick labels from timestamps so panning backward always shows exact dates/times
    tick_step = max(1, n_bars // 10)
    tick_vals = x_positions[::tick_step]
    tick_texts = [
        pd.Timestamp(df.index[i]).strftime("%m-%d %H:%M" if tf_label != "1D" else "%Y-%m-%d")
        for i in tick_vals
    ]

    price_range = max(float(df["High"].max() - df["Low"].min()), 1e-6)
    min_box_h = price_range * 0.015

    # Crisp candlesticks with distinct bullish/bearish colors and wicks
    fig.add_trace(
        go.Candlestick(
            x=x_positions,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=f"{tf_label} Candles",
            increasing=dict(line=dict(color="#26A69A", width=1.2), fillcolor="#26A69A"),
            decreasing=dict(line=dict(color="#EF5350", width=1.2), fillcolor="#EF5350"),
            whiskerwidth=0.35,
        )
    )

    used_label_y: list[float] = []

    def _ok_label(y_val: float) -> bool:
        if any(abs(y_val - py) < price_range * 0.035 for py in used_label_y):
            return False
        used_label_y.append(y_val)
        return True

    # 1. Major S/R Boxes (invalidated by candle close, TF printed on box)
    active_sr = _filter_close_invalidated_sr(df, sr_zones)
    for sz in active_sr[-4:]:
        z_type = str(_get_val(sz, ("zone_type", "kind", "side"), "support")).lower()
        is_sup = "sup" in z_type or "demand" in z_type or "bull" in z_type
        fill_c = "rgba(38, 166, 154, 0.25)" if is_sup else "rgba(239, 83, 80, 0.25)"
        line_c = "rgba(77, 208, 225, 0.95)" if is_sup else "rgba(255, 112, 67, 0.95)"

        pos = _resolve_bar_pos(df, sz, ("start_idx", "idx", "bar_index", "start_bar", "time", "timestamp"))
        raw_b = float(_get_val(sz, ("bottom", "low"), 0.0))
        raw_t = float(_get_val(sz, ("top", "high"), 0.0))
        box_b, box_t = _ensure_min_box_height(raw_b, raw_t, min_box_h)
        z_tf = str(_get_val(sz, ("tf", "timeframe"), tf_label))

        lbl = f"{'SUP' if is_sup else 'RES'} [{z_tf}]"
        _add_zone_box(
            fig=fig,
            x0_pos=pos,
            x1_pos=x_end_pos,
            y_bot=box_b,
            y_top=box_t,
            fill_rgba=fill_c,
            line_rgba=line_c,
            label_text=lbl if _ok_label((box_b + box_t) / 2.0) else f"[{z_tf}]",
            label_color=line_c,
            dash="solid",
        )

    # 2. Minor S/R Boxes (including 15M, TF printed on box)
    if show_minor_sr:
        minor_list = list(minor_sr_zones or []) or _compute_minor_sr_fallback(df, tf_label, min_box_h)
        active_minor = _filter_close_invalidated_sr(df, minor_list)
        for mz in active_minor[-4:]:
            z_type = str(_get_val(mz, ("zone_type", "kind", "side"), "support")).lower()
            is_sup = "sup" in z_type or "demand" in z_type
            m_fill = "rgba(129, 199, 132, 0.18)" if is_sup else "rgba(255, 183, 77, 0.18)"
            m_line = "rgba(165, 214, 167, 0.90)" if is_sup else "rgba(255, 202, 40, 0.90)"

            m_pos = _resolve_bar_pos(df, mz, ("start_idx", "idx", "bar_index", "start_bar", "time", "timestamp"))
            raw_mb = float(_get_val(mz, ("bottom", "low"), 0.0))
            raw_mt = float(_get_val(mz, ("top", "high"), 0.0))
            m_bot, m_top = _ensure_min_box_height(raw_mb, raw_mt, min_box_h)
            m_tf = str(_get_val(mz, ("tf", "timeframe"), tf_label))

            m_lbl = f"Minor {'S' if is_sup else 'R'} [{m_tf}]"
            _add_zone_box(
                fig=fig,
                x0_pos=m_pos,
                x1_pos=x_end_pos,
                y_bot=m_bot,
                y_top=m_top,
                fill_rgba=m_fill,
                line_rgba=m_line,
                label_text=m_lbl if _ok_label((m_bot + m_top) / 2.0) else f"M-[{m_tf}]",
                label_color=m_line,
                dash="dash",
            )

    # 3. OB + FVG Overlap Boxes (sized clearly as boxes with TF label)
    if show_ob_fvg_overlap:
        overlaps = list(paired_zones or []) or _compute_ob_fvg_overlaps_fallback(df, min_box_h)
        for pz in overlaps[-3:]:
            direction = int(_get_val(pz, ("direction",), 1))
            fill_c = "rgba(66, 165, 245, 0.25)" if direction >= 0 else "rgba(171, 71, 188, 0.25)"
            line_c = "rgba(100, 181, 246, 0.95)" if direction >= 0 else "rgba(206, 147, 216, 0.95)"

            ov_low = _get_val(pz, ("overlap_low",))
            ov_high = _get_val(pz, ("overlap_high",))
            if ov_low is not None and ov_high is not None:
                y_low, y_high = _ensure_min_box_height(float(ov_low), float(ov_high), min_box_h * 1.35)
                tag_txt = f"OB+FVG [{tf_label}]"
            else:
                ob_l = float(_get_val(pz, ("ob_low", "bottom"), 0.0))
                ob_h = float(_get_val(pz, ("ob_high", "top"), 0.0))
                fvg_l = float(_get_val(pz, ("fvg_low", "bottom"), ob_l))
                fvg_h = float(_get_val(pz, ("fvg_high", "top"), ob_h))
                y_low, y_high = _ensure_min_box_height(min(ob_l, fvg_l), max(ob_h, fvg_h), min_box_h * 1.35)
                tag_txt = f"OB/FVG [{tf_label}]"

            pos = _resolve_bar_pos(df, pz, ("ob_start_idx", "start_idx", "idx", "bar_index", "time", "timestamp"))
            _add_zone_box(
                fig=fig,
                x0_pos=pos,
                x1_pos=x_end_pos,
                y_bot=y_low,
                y_top=y_high,
                fill_rgba=fill_c,
                line_rgba=line_c,
                label_text=tag_txt if _ok_label((y_low + y_high) / 2.0) else f"[{tf_label}]",
                label_color=line_c,
                dash="dot",
            )

    # 4. Break of Structure (BOS) lines & labels
    if show_bos:
        bos_items: list[tuple[int, int, float, str]] = []
        if sig_df is not None and not sig_df.empty and "bos_level" in sig_df.columns:
            bos_rows = sig_df[sig_df["bos_level"].notna()].tail(4)
            for idx_pos, row in bos_rows.iterrows():
                if idx_pos in df.index:
                    b_pos = int(df.index.get_loc(idx_pos))
                    bos_items.append((b_pos, x_end_pos, float(row["bos_level"]), f"BOS [{tf_label}]"))
        if not bos_items:
            bos_items = [(xs, xb, lvl, f"{lbl} [{tf_label}]") for xs, xb, lvl, lbl in _compute_bos_fallback(df)]

        for x_start, x_brk, lvl, lbl in bos_items[-3:]:
            fig.add_trace(
                go.Scatter(
                    x=[x_start, x_brk],
                    y=[lvl, lvl],
                    mode="lines",
                    line=dict(color="#FFEE58", dash="dash", width=1.4),
                    name="BOS",
                    showlegend=False,
                )
            )
            if _ok_label(lvl):
                fig.add_annotation(
                    xref="x",
                    yref="y",
                    x=x_brk,
                    y=lvl,
                    text=lbl,
                    showarrow=False,
                    xanchor="right",
                    yanchor="bottom",
                    font=dict(size=10, color="#FFEE58"),
                )

    x_start_view = max(0, n_bars - int(initial_visible_bars))
    fig.update_layout(
        title=title,
        template="plotly_dark",
        dragmode="pan",
        xaxis_rangeslider_visible=False,
        height=580,
        margin=dict(l=10, r=110, t=42, b=20),
        xaxis=dict(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_texts,
            range=[x_start_view, x_end_pos + 6],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
        ),
    )
    return fig


def render_worker_panel() -> None:
    st.subheader("24/7 Cloud Worker Feed (`data/worker_state.json`)")
    state_path = Path("data/worker_state.json")
    if not state_path.exists():
        st.info("Worker state file not found yet. Run `python worker.py` in a background worker service.")
        return

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.warning(f"Unable to parse worker state: {exc}")
        return

    st.caption(f"Status: `{state.get('status', 'unknown')}` | Last Update (UTC): `{state.get('updated_utc', 'n/a')}`")
    signals = state.get("signals", [])
    if not signals:
        st.write("No scalp setups logged in worker history yet.")
        return

    for s in reversed(signals[-6:]):
        st.markdown(
            f"**{s.get('asset')} ({s.get('symbol')})** — `{s.get('direction')}` ({s.get('confidence')}) | "
            f"Entry: `{s.get('entry')}` | SL: `{s.get('sl')}` | TP: `{s.get('tp')}` (`1:{s.get('rr')}` R:R) | "
            f"Exec: `{s.get('execution_status', 'n/a')}`"
        )


def main() -> None:
    if not render_auth_gate():
        return

    user = st.session_state["auth_user"]
    acct: VirtualAccount = st.session_state.setdefault("virtual_account", VirtualAccount())
    broker: BrokerSession = st.session_state.setdefault("broker_session", BrokerSession())

    with st.sidebar:
        st.write(f"Signed in as **{user}**")
        if st.button("Sign Out"):
            st.session_state.pop("auth_user", None)
            if hasattr(st, "logout"):
                try:
                    st.logout()
                except Exception:
                    pass
            st.rerun()

        st.divider()
        asset_name = st.selectbox("Market (Forex, Commodities, Crypto)", list(WATCHLIST.keys()), index=0)
        symbol = WATCHLIST[asset_name]
        tf_label = st.selectbox("Chart Timeframe", list(TF_CONFIG.keys()), index=3)  # Default 15M
        visible_bars = st.slider("Visible Candles Window", min_value=60, max_value=400, value=120, step=20)
        rr_target = st.slider("Target R:R", min_value=2.0, max_value=15.0, value=5.0, step=0.5)
        risk_pct = st.slider("Risk per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)

        st.divider()
        st.caption("Chart Layers")
        show_minor_sr = st.checkbox("Minor S/R Boxes (incl. 15M)", value=True)
        show_bos = st.checkbox("Break of Structure (BOS)", value=True)
        show_ob_fvg_overlap = st.checkbox("OB + FVG Overlap Boxes", value=True)

    st.title(f"DESTINY AI — {asset_name} ({symbol})")

    df = fetch_tf_chart(symbol, tf_label, lookback_bars=500)
    tf_mins = {"1D": 1440, "4H": 240, "1H": 60, "15M": 15, "5M": 5, "1M": 1}.get(tf_label, 15)
    sr_out = run_sr_zones_engine(df, invalidation_mode="Close")
    ob_out = run_ob_fvg_engine(df, timeframe_minutes=tf_mins)

    minor_sr = (
        sr_out.get("minor_zones")
        or sr_out.get("minor_sr")
        or sr_out.get("secondary_zones")
        or []
    )
    overlap_zones = (
        ob_out.get("overlaps")
        or ob_out.get("overlapping_zones")
        or ob_out.get("active_zones")
        or []
    )

    chart_col, side_col = st.columns([2.2, 1.0])

    with chart_col:
        fig = build_chart(
            df=df,
            title=f"{asset_name} ({symbol}) — {tf_label}",
            tf_label=tf_label,
            sr_zones=sr_out.get("active_zones", []),
            minor_sr_zones=minor_sr,
            paired_zones=overlap_zones,
            sig_df=ob_out.get("signals_df", pd.DataFrame()),
            show_minor_sr=show_minor_sr,
            show_bos=show_bos,
            show_ob_fvg_overlap=show_ob_fvg_overlap,
            initial_visible_bars=visible_bars,
        )
        st.plotly_chart(fig, use_container_width=True)

    with side_col:
        st.subheader("Virtual Account")
        current_px = float(df["Close"].iloc[-1]) if not df.empty else 0.0
        unrl = acct.unrealized_pnl({symbol: current_px}) if current_px > 0 else 0.0
        st.metric("Cash Balance", f"${acct.balance:,.2f}", delta=f"Unrealized: {unrl:+,.2f}")

        if current_px > 0:
            side = st.selectbox("Order Side", ["LONG", "SHORT"])
            default_sl = current_px * (0.995 if side == "LONG" else 1.005)
            default_tp = current_px + (abs(current_px - default_sl) * rr_target) * (1 if side == "LONG" else -1)
            entry_in = st.number_input("Entry Price", value=float(current_px), format="%.5f")
            sl_in = st.number_input("Stop Loss", value=float(default_sl), format="%.5f")
            tp_in = st.number_input("Take Profit", value=float(default_tp), format="%.5f")
            if st.button("Place Virtual Order"):
                ok, msg = acct.open_position_with_risk(symbol, side, entry_in, sl_in, tp_in, risk_pct=risk_pct)
                (st.success if ok else st.error)(msg)

        if acct.positions:
            st.write(f"Open Positions: {len(acct.positions)}")
            for i, pos in enumerate(acct.positions):
                st.caption(f"#{i+1} {pos.side} {pos.symbol} @ {pos.entry_price:.5f} | SL {pos.stop_loss:.5f} | TP {pos.take_profit:.5f}")

    st.divider()
    render_worker_panel()

    with st.expander("Broker Session Status"):
        if broker.connected:
            st.write(f"Connected to **{broker.account_label}** ({broker.environment}) — Key `{broker.api_key_masked}`")
        else:
            st.write("No active broker session connected (Virtual Mode active).")


if __name__ == "__main__":
    main()
