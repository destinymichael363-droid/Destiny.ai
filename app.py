from __future__ import annotations

import json
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
    "Bitcoin": "BTC-USD",
    "Gold": "GC=F",
    "Crude Oil": "CL=F",
    "EUR/USD": "EURUSD=X",
    "GBP/JPY": "GBPJPY=X",
}

TF_CONFIG: dict[str, tuple[str, str]] = {
    "1H": ("1h", "60d"),
    "15M": ("15m", "30d"),
    "5M": ("5m", "14d"),
    "1M": ("1m", "5d"),
}


def _clean_ohlcv(df: pd.DataFrame | None) -> pd.DataFrame:
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    if not all(c in out.columns for c in ["Open", "High", "Low", "Close"]):
        return pd.DataFrame(columns=cols)
    if "Volume" not in out.columns:
        out["Volume"] = 0.0
    out = out[cols].dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    return out[~out.index.duplicated(keep="last")]


@st.cache_data(ttl=45)
def fetch_tf_chart(symbol: str, tf_label: str) -> pd.DataFrame:
    interval, period = TF_CONFIG.get(tf_label, ("1h", "60d"))
    try:
        raw = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False, actions=False)
        return _clean_ohlcv(raw).tail(220)
    except Exception:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def render_auth_gate() -> bool:
    if st.session_state.get("auth_user"):
        return True

    st.title("DESTINY AI — Sign In or Create Account")
    st.caption("Private write access protects individual virtual accounts and session state.")

    tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])

    with tab_login:
        with st.form("login_form"):
            ident = st.text_input("Username or Email")
            pw = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In")
            if submitted:
                res = authenticate_user(ident, pw)
                if res.ok and res.username:
                    st.session_state["auth_user"] = res.username
                    st.session_state["virtual_account"] = VirtualAccount()
                    st.session_state["broker_session"] = BrokerSession()
                    st.rerun()
                else:
                    st.error(res.message)

    with tab_signup:
        with st.form("signup_form"):
            u = st.text_input("Username (3–24 letters, numbers, underscores)")
            em = st.text_input("Email")
            pw1 = st.text_input("Password (8+ characters)", type="password")
            pw2 = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Create Account")
            if submitted:
                if pw1 != pw2:
                    st.error("Passwords do not match.")
                else:
                    res = register_user(u, em, pw1)
                    if res.ok and res.username:
                        st.session_state["auth_user"] = res.username
                        st.session_state["virtual_account"] = VirtualAccount()
                        st.session_state["broker_session"] = BrokerSession()
                        st.success("Account created. Welcome!")
                        st.rerun()
                    else:
                        st.error(res.message)

    return False


def _resolve_x_index(df: pd.DataFrame, obj: object, attr_names: tuple[str, ...]) -> int:
    if df.empty:
        return 0
    raw_val = None
    for name in attr_names:
        if isinstance(obj, dict) and name in obj:
            raw_val = obj[name]
            break
        if hasattr(obj, name):
            raw_val = getattr(obj, name)
            break
    if raw_val is None:
        return 0
    if isinstance(raw_val, (int, float, np.integer, np.floating)) and not isinstance(raw_val, bool):
        return max(0, min(int(raw_val), len(df) - 1))
    try:
        idx_loc = df.index.get_indexer([raw_val], method="nearest")[0]
        if idx_loc >= 0:
            return max(0, min(int(idx_loc), len(df) - 1))
    except Exception:
        pass
    return 0


def _get_val(obj: object, names: tuple[str, ...], default: object = None) -> object:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _filter_close_invalidated_sr(df: pd.DataFrame, zones: list) -> list:
    """Removes S/R zones whose range is broken by a candle CLOSE after zone start."""
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

        # Respect explicit invalidated flag if engine already marked it
        if bool(_get_val(sz, ("invalidated", "broken", "mitigated"), False)):
            continue

        start_pos = _resolve_x_index(df, sz, ("start_idx", "idx", "bar_index", "start_bar", "time", "timestamp"))
        sub_closes = closes[start_pos:]
        # Close-based invalidation: a candle closes beyond the zone boundary
        if np.any(sub_closes > t_val) or np.any(sub_closes < b_val):
            continue
        valid_zones.append(sz)
    return valid_zones


def _compute_minor_sr_fallback(df: pd.DataFrame, tf_label: str) -> list[dict]:
    """Local swing minor S/R levels (especially helpful on 15M) when engine returns few/none."""
    if len(df) < 12:
        return []
    lookback = 3 if tf_label == "15M" else 4
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    atr_proxy = float(np.median(highs - lows)) * 0.35
    pad = max(atr_proxy, 1e-6)

    levels: list[dict] = []
    for i in range(lookback, len(df) - lookback):
        if highs[i] == np.max(highs[i - lookback : i + lookback + 1]):
            px = float(highs[i])
            levels.append({
                "zone_type": "resistance",
                "tier": "minor",
                "start_idx": i,
                "bottom": px - pad,
                "top": px + pad,
            })
        if lows[i] == np.min(lows[i - lookback : i + lookback + 1]):
            px = float(lows[i])
            levels.append({
                "zone_type": "support",
                "tier": "minor",
                "start_idx": i,
                "bottom": px - pad,
                "top": px + pad,
            })
    return levels[-6:]


def _compute_ob_fvg_overlaps_fallback(df: pd.DataFrame) -> list[dict]:
    """Detects 3-candle FVGs and nearby impulse Order Blocks with overlapping price ranges."""
    if len(df) < 5:
        return []
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    opens = df["Open"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)

    fvgs: list[tuple[int, int, float, float]] = []
    obs: list[tuple[int, int, float, float]] = []

    for i in range(1, len(df) - 1):
        # Bullish FVG: low[i+1] > high[i-1]
        if lows[i + 1] > highs[i - 1]:
            fvgs.append((1, i, float(highs[i - 1]), float(lows[i + 1])))
        # Bearish FVG: high[i+1] < low[i-1]
        if highs[i + 1] < lows[i - 1]:
            fvgs.append((-1, i, float(highs[i + 1]), float(lows[i - 1])))

        # Bullish OB: down candle followed by strong up candle
        if closes[i - 1] < opens[i - 1] and closes[i] > opens[i] and closes[i] > highs[i - 1]:
            obs.append((1, i - 1, float(lows[i - 1]), float(highs[i - 1])))
        # Bearish OB: up candle followed by strong down candle
        if closes[i - 1] > opens[i - 1] and closes[i] < opens[i] and closes[i] < lows[i - 1]:
            obs.append((-1, i - 1, float(lows[i - 1]), float(highs[i - 1])))

    overlaps: list[dict] = []
    for f_dir, f_idx, f_bot, f_top in fvgs[-12:]:
        for o_dir, o_idx, o_bot, o_top in obs[-12:]:
            if f_dir == o_dir and abs(f_idx - o_idx) <= 8:
                ov_bot = max(f_bot, o_bot)
                ov_top = min(f_top, o_top)
                if ov_top > ov_bot:
                    overlaps.append({
                        "direction": f_dir,
                        "ob_start_idx": min(f_idx, o_idx),
                        "ob_low": o_bot,
                        "ob_high": o_top,
                        "fvg_low": f_bot,
                        "fvg_high": f_top,
                        "overlap_low": ov_bot,
                        "overlap_high": ov_top,
                    })
    return overlaps[-5:]


def _compute_bos_fallback(df: pd.DataFrame) -> list[tuple[object, object, float, str]]:
    """Returns (x0, x1, price, label) for recent Break of Structure events on candle close."""
    if len(df) < 10:
        return []
    swing_n = 3
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    idx = df.index

    last_swing_high: tuple[int, float] | None = None
    last_swing_low: tuple[int, float] | None = None
    bos_events: list[tuple[object, object, float, str]] = []

    for i in range(swing_n, len(df) - swing_n):
        if highs[i] == np.max(highs[i - swing_n : i + swing_n + 1]):
            last_swing_high = (i, float(highs[i]))
        if lows[i] == np.min(lows[i - swing_n : i + swing_n + 1]):
            last_swing_low = (i, float(lows[i]))

    for i in range(swing_n + 1, len(df)):
        if last_swing_high is not None and i > last_swing_high[0] and closes[i] > last_swing_high[1]:
            bos_events.append((idx[last_swing_high[0]], idx[i], last_swing_high[1], "BOS ↑"))
            last_swing_high = None
        if last_swing_low is not None and i > last_swing_low[0] and closes[i] < last_swing_low[1]:
            bos_events.append((idx[last_swing_low[0]], idx[i], last_swing_low[1], "BOS ↓"))
            last_swing_low = None

    return bos_events[-6:]


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
) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title=f"{title} (No data)", template="plotly_dark", height=540)
        return fig

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        )
    )

    x_end = df.index[-1]

    # 1. Major / Active S&R zones (invalidated strictly by candle close)
    active_sr = _filter_close_invalidated_sr(df, sr_zones)
    for sz in active_sr[-6:]:
        zone_type = str(_get_val(sz, ("zone_type", "kind", "side"), "support")).lower()
        is_sup = "sup" in zone_type or "demand" in zone_type or "bull" in zone_type
        fill_c = "rgba(46, 204, 113, 0.16)" if is_sup else "rgba(231, 76, 60, 0.16)"
        line_c = "rgba(46, 204, 113, 0.85)" if is_sup else "rgba(231, 76, 60, 0.85)"
        pos = _resolve_x_index(df, sz, ("start_idx", "idx", "bar_index", "start_bar", "time", "timestamp"))
        fig.add_shape(
            type="rect",
            x0=df.index[pos],
            x1=x_end,
            y0=float(_get_val(sz, ("bottom", "low"), 0.0)),
            y1=float(_get_val(sz, ("top", "high"), 0.0)),
            fillcolor=fill_c,
            line=dict(color=line_c, width=1.2),
        )

    # 2. Minor S&R zones (always includes 15M minor levels when enabled)
    if show_minor_sr:
        combined_minor = list(minor_sr_zones or [])
        if not combined_minor:
            combined_minor = _compute_minor_sr_fallback(df, tf_label)
        active_minor = _filter_close_invalidated_sr(df, combined_minor)
        for mz in active_minor[-6:]:
            z_type = str(_get_val(mz, ("zone_type", "kind", "side"), "support")).lower()
            is_sup = "sup" in z_type or "demand" in z_type
            m_fill = "rgba(88, 214, 141, 0.10)" if is_sup else "rgba(241, 148, 138, 0.10)"
            m_line = "rgba(88, 214, 141, 0.65)" if is_sup else "rgba(241, 148, 138, 0.65)"
            m_pos = _resolve_x_index(df, mz, ("start_idx", "idx", "bar_index", "start_bar", "time", "timestamp"))
            m_bot = float(_get_val(mz, ("bottom", "low"), 0.0))
            m_top = float(_get_val(mz, ("top", "high"), 0.0))
            fig.add_shape(
                type="rect",
                x0=df.index[m_pos],
                x1=x_end,
                y0=m_bot,
                y1=m_top,
                fillcolor=m_fill,
                line=dict(color=m_line, width=1, dash="dash"),
            )
            fig.add_annotation(
                x=x_end,
                y=(m_bot + m_top) / 2.0,
                text=f"Minor {'S' if is_sup else 'R'} ({tf_label})",
                showarrow=False,
                xanchor="left",
                font=dict(size=10, color=m_line),
            )

    # 3. OB + FVG Overlap zones
    if show_ob_fvg_overlap:
        overlaps = list(paired_zones or [])
        if not overlaps:
            overlaps = _compute_ob_fvg_overlaps_fallback(df)
        for pz in overlaps[-5:]:
            direction = int(_get_val(pz, ("direction",), 1))
            fill_c = "rgba(52, 152, 219, 0.20)" if direction >= 0 else "rgba(155, 89, 182, 0.20)"
            line_c = "rgba(93, 173, 226, 0.9)" if direction >= 0 else "rgba(187, 143, 206, 0.9)"

            ov_low = _get_val(pz, ("overlap_low",))
            ov_high = _get_val(pz, ("overlap_high",))
            if ov_low is not None and ov_high is not None and float(ov_high) > float(ov_low):
                y_low, y_high = float(ov_low), float(ov_high)
                tag_txt = "OB+FVG Overlap"
            else:
                ob_l = float(_get_val(pz, ("ob_low", "bottom"), 0.0))
                ob_h = float(_get_val(pz, ("ob_high", "top"), 0.0))
                fvg_l = float(_get_val(pz, ("fvg_low", "bottom"), ob_l))
                fvg_h = float(_get_val(pz, ("fvg_high", "top"), ob_h))
                y_low, y_high = min(ob_l, fvg_l), max(ob_h, fvg_h)
                tag_txt = "OB/FVG Zone"

            pos = _resolve_x_index(df, pz, ("ob_start_idx", "start_idx", "idx", "bar_index", "time", "timestamp"))
            fig.add_shape(
                type="rect",
                x0=df.index[pos],
                x1=x_end,
                y0=y_low,
                y1=y_high,
                fillcolor=fill_c,
                line=dict(color=line_c, width=1.2, dash="dot"),
            )
            fig.add_annotation(
                x=df.index[pos],
                y=y_high,
                text=tag_txt,
                showarrow=False,
                yshift=10,
                font=dict(size=10, color=line_c),
            )

    # 4. Break of Structure (BOS) lines & labels
    if show_bos:
        bos_plotted = 0
        if sig_df is not None and not sig_df.empty and "bos_level" in sig_df.columns:
            bos_rows = sig_df[sig_df["bos_level"].notna()].tail(6)
            for idx_pos, row in bos_rows.iterrows():
                if idx_pos in df.index:
                    lvl = float(row["bos_level"])
                    fig.add_trace(
                        go.Scatter(
                            x=[idx_pos, x_end],
                            y=[lvl, lvl],
                            mode="lines+text",
                            text=["", "BOS"],
                            textposition="top right",
                            line=dict(color="#F5B041", dash="dash", width=1.2),
                            name="BOS",
                            showlegend=False,
                        )
                    )
                    bos_plotted += 1

        if bos_plotted == 0:
            for x_start, x_brk, lvl, lbl in _compute_bos_fallback(df):
                fig.add_trace(
                    go.Scatter(
                        x=[x_start, x_brk],
                        y=[lvl, lvl],
                        mode="lines+text",
                        text=["", lbl],
                        textposition="top right",
                        line=dict(color="#F5B041", dash="dash", width=1.2),
                        name="BOS",
                        showlegend=False,
                    )
                )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=550,
        margin=dict(l=10, r=70, t=42, b=10),
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
            st.rerun()

        st.divider()
        asset_name = st.selectbox("Market", list(WATCHLIST.keys()), index=0)
        symbol = WATCHLIST[asset_name]
        tf_label = st.selectbox("Chart Timeframe", list(TF_CONFIG.keys()), index=1 if "15M" in TF_CONFIG else 0)
        rr_target = st.slider("Target R:R", min_value=2.0, max_value=15.0, value=5.0, step=0.5)
        risk_pct = st.slider("Risk per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)

        st.divider()
        st.caption("Chart Layers")
        show_minor_sr = st.checkbox("Minor S/R Levels (incl. 15M)", value=True)
        show_bos = st.checkbox("Break of Structure (BOS)", value=True)
        show_ob_fvg_overlap = st.checkbox("OB + FVG Overlaps", value=True)

    st.title(f"DESTINY AI — {asset_name} ({symbol})")

    df = fetch_tf_chart(symbol, tf_label)
    sr_out = run_sr_zones_engine(df, invalidation_mode="Close")
    ob_out = run_ob_fvg_engine(df, timeframe_minutes=60 if tf_label == "1H" else 15)

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
