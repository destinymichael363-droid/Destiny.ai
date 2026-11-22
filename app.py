from __future__ import annotations

import json
from pathlib import Path

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
        return _clean_ohlcv(raw).tail(180)
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
        if hasattr(obj, name):
            raw_val = getattr(obj, name)
            break

    if raw_val is None:
        return 0

    if isinstance(raw_val, (int, float)) and not isinstance(raw_val, bool):
        pos = int(raw_val)
        return max(0, min(pos, len(df) - 1))

    try:
        idx_loc = df.index.get_indexer([raw_val], method="nearest")[0]
        if idx_loc >= 0:
            return max(0, min(int(idx_loc), len(df) - 1))
    except Exception:
        pass

    return 0


def build_chart(
    df: pd.DataFrame,
    title: str,
    sr_zones: list,
    paired_zones: list,
    sig_df: pd.DataFrame,
) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title=f"{title} (No data)", template="plotly_dark", height=500)
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

    for sz in (sr_zones or [])[-6:]:
        zone_type = str(getattr(sz, "zone_type", "support")).lower()
        is_support = zone_type == "support"
        fill_c = "rgba(46, 204, 113, 0.16)" if is_support else "rgba(231, 76, 60, 0.16)"
        line_c = "rgba(46, 204, 113, 0.75)" if is_support else "rgba(231, 76, 60, 0.75)"

        pos = _resolve_x_index(df, sz, ("start_idx", "idx", "bar_index", "start_bar", "time", "timestamp"))
        fig.add_shape(
            type="rect",
            x0=df.index[pos],
            x1=x_end,
            y0=float(sz.bottom),
            y1=float(sz.top),
            fillcolor=fill_c,
            line=dict(color=line_c, width=1),
        )

    for pz in (paired_zones or [])[-6:]:
        direction = int(getattr(pz, "direction", 1))
        fill_c = "rgba(52, 152, 219, 0.16)" if direction == 1 else "rgba(155, 89, 182, 0.16)"
        line_c = "rgba(52, 152, 219, 0.75)" if direction == 1 else "rgba(155, 89, 182, 0.75)"

        y_low = min(float(pz.ob_low), float(pz.fvg_low))
        y_high = max(float(pz.ob_high), float(pz.fvg_high))
        pos = _resolve_x_index(df, pz, ("ob_start_idx", "start_idx", "idx", "bar_index", "time", "timestamp"))

        fig.add_shape(
            type="rect",
            x0=df.index[pos],
            x1=x_end,
            y0=y_low,
            y1=y_high,
            fillcolor=fill_c,
            line=dict(color=line_c, width=1, dash="dot"),
        )

    if sig_df is not None and not sig_df.empty and "bos_level" in sig_df.columns:
        bos_rows = sig_df[sig_df["bos_level"].notna()].tail(6)
        for idx_pos, row in bos_rows.iterrows():
            if idx_pos in df.index:
                fig.add_trace(
                    go.Scatter(
                        x=[idx_pos, x_end],
                        y=[float(row["bos_level"]), float(row["bos_level"])],
                        mode="lines",
                        line=dict(dash="dash", width=1),
                        name="BOS",
                        showlegend=False,
                    )
                )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
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
        tf_label = st.selectbox("Chart Timeframe", list(TF_CONFIG.keys()), index=0)
        rr_target = st.slider("Target R:R", min_value=2.0, max_value=15.0, value=5.0, step=0.5)
        risk_pct = st.slider("Risk per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)

    st.title(f"DESTINY AI — {asset_name} ({symbol})")

    df = fetch_tf_chart(symbol, tf_label)
    sr_out = run_sr_zones_engine(df, invalidation_mode="Wick")
    ob_out = run_ob_fvg_engine(df, timeframe_minutes=60 if tf_label == "1H" else 15)

    chart_col, side_col = st.columns([2.2, 1.0])

    with chart_col:
        fig = build_chart(
            df=df,
            title=f"{asset_name} ({symbol}) — {tf_label}",
            sr_zones=sr_out.get("active_zones", []),
            paired_zones=ob_out.get("active_zones", []),
            sig_df=ob_out.get("signals_df", pd.DataFrame()),
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
