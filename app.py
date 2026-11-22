from __future__ import annotations

import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from auth_and_store import authenticate_user, register_user
from broker_and_account import BrokerSession, VirtualAccount
from mtf_runner import run_closed_candle_5tf_cascade
from notifications import NotificationCenter
from ob_fvg_engine import run_ob_fvg_engine
from sr_engine import run_sr_zones_engine


# =========================================================
# DESTINY AI — PUBLIC 5-TF SCALP, LIQUIDITY & ACCOUNT APP
# =========================================================

st.set_page_config(
    page_title="DESTINY AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# MARKET & TIMEFRAME CONFIG
# =========================================================

MARKETS = {
    "Crypto": {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "Solana": "SOL-USD",
        "XRP": "XRP-USD",
        "BNB": "BNB-USD",
        "Litecoin": "LTC-USD",
        "Dogecoin": "DOGE-USD",
    },
    "Forex": {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "USDJPY=X",
        "GBP/JPY": "GBPJPY=X",
        "EUR/JPY": "EURJPY=X",
        "AUD/USD": "AUDUSD=X",
        "USD/CAD": "USDCAD=X",
        "USD/CHF": "USDCHF=X",
        "NZD/USD": "NZDUSD=X",
    },
    "Commodities": {
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Crude Oil": "CL=F",
        "Brent Oil": "BZ=F",
        "Natural Gas": "NG=F",
        "Platinum": "PL=F",
        "Copper": "HG=F",
    },
}

TIMEFRAMES = ["4H", "1H", "15M", "5M", "1M", "1D"]

TF_MINUTES = {
    "1M": 1,
    "5M": 5,
    "15M": 15,
    "1H": 60,
    "4H": 240,
}

OHLCV = ["Open", "High", "Low", "Close", "Volume"]

DATA_SETTINGS = {
    "1D": ("1d", "2y"),
    "1H": ("1h", "730d"),
    "15M": ("15m", "60d"),
    "5M": ("5m", "60d"),
    "1M": ("1m", "7d"),
}


# =========================================================
# SESSION STATE & PER-USER ACCOUNT ISOLATION
# =========================================================

if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "user_accounts" not in st.session_state:
    st.session_state.user_accounts = {}

if "user_brokers" not in st.session_state:
    st.session_state.user_brokers = {}

if "account_mode" not in st.session_state:
    st.session_state.account_mode = "Virtual Account"

if "notifier" not in st.session_state:
    st.session_state.notifier = NotificationCenter()

if "tg_token" not in st.session_state:
    st.session_state.tg_token = ""

if "tg_chat_id" not in st.session_state:
    st.session_state.tg_chat_id = ""


def get_user_virtual_account(username: str) -> VirtualAccount:
    accounts: dict[str, VirtualAccount] = st.session_state.user_accounts
    if username not in accounts:
        accounts[username] = VirtualAccount(balance=10_000.0, starting_balance=10_000.0)
    return accounts[username]


def get_user_broker_session(username: str) -> BrokerSession:
    brokers: dict[str, BrokerSession] = st.session_state.user_brokers
    if username not in brokers:
        brokers[username] = BrokerSession()
    return brokers[username]


# =========================================================
# DATA LOADING & CLEANING
# =========================================================

def empty_data() -> pd.DataFrame:
    return pd.DataFrame(columns=OHLCV)


def clean_market_data(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return empty_data()

    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]
    if not all(col in frame.columns for col in required):
        return empty_data()

    if "Volume" not in frame.columns:
        frame["Volume"] = 0.0

    frame = frame[OHLCV].copy()
    for col in OHLCV:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.dropna(subset=required).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]

    valid = (
        (frame["High"] >= frame["Open"])
        & (frame["High"] >= frame["Close"])
        & (frame["High"] >= frame["Low"])
        & (frame["Low"] <= frame["Open"])
        & (frame["Low"] <= frame["Close"])
        & (frame["Low"] <= frame["High"])
    )
    return frame.loc[valid]


@st.cache_data(ttl=180, show_spinner=False)
def fetch_market_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    try:
        ticker = yf.Ticker(symbol)
        frame = ticker.history(
            period=period,
            interval=interval,
            auto_adjust=False,
            actions=False,
            prepost=False,
        )
        return clean_market_data(frame)
    except Exception:
        return empty_data()


def create_four_hour_data(hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly is None or hourly.empty:
        return empty_data()
    try:
        result = hourly.resample("4h", label="right", closed="right").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })
        return clean_market_data(result)
    except Exception:
        return empty_data()


def load_timeframe(symbol: str, timeframe: str) -> pd.DataFrame:
    if timeframe == "4H":
        hourly = fetch_market_data(symbol, "1h", "730d")
        return create_four_hour_data(hourly)
    interval, period = DATA_SETTINGS[timeframe]
    return fetch_market_data(symbol, interval, period)


def load_all_5_timeframes(symbol: str) -> dict[str, pd.DataFrame]:
    hourly = fetch_market_data(symbol, "1h", "730d")
    return {
        "4H": create_four_hour_data(hourly),
        "1H": hourly,
        "15M": fetch_market_data(symbol, "15m", "60d"),
        "5M": fetch_market_data(symbol, "5m", "60d"),
        "1M": fetch_market_data(symbol, "1m", "7d"),
    }


# =========================================================
# AUTHENTICATION & ACCOUNT PROTECTION BAR
# =========================================================

def display_auth_section() -> None:
    st.subheader("Account Access & Security")

    current_user = st.session_state.current_user
    if current_user:
        auth_col, logout_col = st.columns([4, 1])
        with auth_col:
            st.success(
                f"Signed in as **`{current_user}`** — Virtual balance, open trades, "
                "and broker session state are isolated to your account."
            )
        with logout_col:
            if st.button("Log Out", use_container_width=True):
                st.session_state.notifier.push(
                    event_id=f"logout_{current_user}_{pd.Timestamp.utcnow().strftime('%Y%m%d%H%M')}",
                    category="SECURITY",
                    title="Session Logged Out",
                    detail=f"User '{current_user}' logged out cleanly.",
                )
                st.session_state.current_user = None
                st.rerun()
        return

    st.info(
        "Public preview mode is active: you can inspect charts, liquidity sweeps, and the 5-TF cascade. "
        "Sign in or create an account below to use your personal Virtual Account or connect a Broker."
    )

    tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])

    with tab_login:
        with st.form("user_login_form", clear_on_submit=False):
            login_id = st.text_input("Username or Email", key="login_id")
            login_pw = st.text_input("Password", type="password", key="login_pw")
            login_sub = st.form_submit_button("Log In")

            if login_sub:
                res = authenticate_user(login_id, login_pw)
                if res.ok and res.username:
                    st.session_state.current_user = res.username
                    get_user_virtual_account(res.username)
                    get_user_broker_session(res.username)
                    st.session_state.notifier.push(
                        event_id=f"login_{res.username}_{pd.Timestamp.utcnow().strftime('%Y%m%d%H%M')}",
                        category="SECURITY",
                        title="Account Login",
                        detail=f"Signed in to '{res.username}'.",
                    )
                    st.success(res.message)
                    st.rerun()
                else:
                    st.error(res.message)

    with tab_signup:
        with st.form("user_signup_form", clear_on_submit=True):
            su_user = st.text_input("Choose Username (3–24 chars, letters/numbers/_)", key="su_user")
            su_email = st.text_input("Email Address", key="su_email")
            su_pw = st.text_input("Create Password (min 8 characters)", type="password", key="su_pw")
            su_pw2 = st.text_input("Confirm Password", type="password", key="su_pw2")
            su_sub = st.form_submit_button("Create Account")

            if su_sub:
                if su_pw != su_pw2:
                    st.error("Passwords do not match.")
                else:
                    res = register_user(su_user, su_email, su_pw)
                    if res.ok:
                        st.success(res.message)
                    else:
                        st.error(res.message)


# =========================================================
# NOTIFICATIONS PANEL (IN-APP + OPTIONAL TELEGRAM)
# =========================================================

def display_notifications_panel() -> None:
    st.subheader("Notifications & Alert Feed")

    notifier: NotificationCenter = st.session_state.notifier

    with st.expander("Telegram Alert Settings (Optional)", expanded=False):
        t1, t2 = st.columns(2)
        with t1:
            st.session_state.tg_token = st.text_input(
                "Telegram Bot Token",
                value=st.session_state.tg_token,
                type="password",
                placeholder="Optional: 123456:ABC-DEF...",
            )
        with t2:
            st.session_state.tg_chat_id = st.text_input(
                "Telegram Chat ID",
                value=st.session_state.tg_chat_id,
                placeholder="Optional: your chat or channel ID",
            )

    if notifier.items:
        for item in notifier.items[:8]:
            st.write(
                f"**[{item.category}] {item.title}** — {item.detail} *({item.timestamp})*"
            )
    else:
        st.caption("No alerts triggered yet. Scalp signals, liquidity sweeps, and TP/SL hits will appear here once per event.")


# =========================================================
# 5-TIMEFRAME CASCADE DISPLAY
# =========================================================

def display_5tf_cascade(mtf_out: dict, symbol: str) -> None:
    st.subheader("Closed-Candle 5-Timeframe Cascade (4H → 1H → 15M → 5M → 1M)")

    trends = mtf_out["tf_trends"]
    lbl = {1: "BULL", -1: "BEAR", 0: "NEUTRAL"}

    cols = st.columns(5)
    for idx, tf in enumerate(["4H", "1H", "15M", "5M", "1M"]):
        cols[idx].metric(f"{tf} Trend", lbl.get(trends.get(tf, 0), "NEUTRAL"))

    sweeps = mtf_out["recent_sweeps"]
    st.caption(
        f"HTF Bias: **{lbl.get(mtf_out['htf_bias'], 'NEUTRAL')}** | "
        f"HTF S/R ↔ LTF OB+FVG Overlaps: **{len(mtf_out['overlaps'])}** | "
        f"Recent Liquidity Sweeps — 15M: `{sweeps['15M']}`, 5M: `{sweeps['5M']}`, 1M: `{sweeps['1M']}`"
    )

    if mtf_out["overlaps"]:
        overlap_rows = [
            {
                "Direction": "LONG" if ov["direction"] == 1 else "SHORT",
                "HTF Anchor Zone": ov["htf_source"],
                "LTF Overlapping OB+FVG": ov["ltf_source"],
            }
            for ov in mtf_out["overlaps"][:6]
        ]
        st.dataframe(pd.DataFrame(overlap_rows), use_container_width=True, height=180)

    sig = mtf_out.get("scalp_signal")
    if sig is not None:
        st.success(
            f"**ACTIVE {sig.direction} SCALP SETUP ({sig.confidence})** — "
            f"Entry: `{sig.entry_price:,.5f}` | SL: `{sig.stop_loss:,.5f}` | "
            f"TP: `{sig.take_profit:,.5f}` (`1:{sig.rr_ratio:.1f} R:R`)\n\n"
            f"HTF Anchor: {sig.htf_zone_summary} → LTF Overlap: {sig.ltf_overlap_summary} ({sig.sweep_summary})"
        )
        if st.session_state.current_user:
            if st.button("Load Scalp Signal into Order Ticket", key=f"load_sig_{symbol}"):
                st.session_state[f"ord_side_{symbol}"] = sig.direction
                st.session_state[f"ord_sl_{symbol}"] = float(sig.stop_loss)
                st.session_state[f"ord_tp_{symbol}"] = float(sig.take_profit)
                st.rerun()
        else:
            st.caption("Log in above to load this signal into your personal order ticket.")
    else:
        st.info("No aligned 4H→1M scalp signal on closed candles right now. Waiting for HTF zone overlap + 1M trigger.")


# =========================================================
# ACCOUNT, SL/TP RISK & BROKER PANEL (PROTECTED BY LOGIN)
# =========================================================

def display_account_panel(symbol: str, latest_price: float, atr_val: float) -> None:
    st.subheader("Account, Risk & Execution")

    current_user = st.session_state.current_user
    if not current_user:
        st.warning("Account execution and broker credentials are locked until you log in above.")
        return

    chosen_mode = st.radio(
        "Active Account Mode",
        options=["Virtual Account", "Real Broker Account"],
        index=0 if st.session_state.account_mode == "Virtual Account" else 1,
        horizontal=True,
    )
    st.session_state.account_mode = chosen_mode

    va: VirtualAccount = get_user_virtual_account(current_user)
    bs: BrokerSession = get_user_broker_session(current_user)

    if chosen_mode == "Virtual Account":
        upnl = va.unrealized_pnl({symbol: latest_price})
        equity = va.balance + upnl

        m1, m2, m3 = st.columns(3)
        m1.metric("Virtual Cash", f"${va.balance:,.2f}")
        m2.metric("Unrealized P&L", f"${upnl:+,.2f}")
        m3.metric("Estimated Equity", f"${equity:,.2f}")

        default_sl = max(0.00001, round(latest_price - (atr_val * 1.5), 5))
        default_tp = round(latest_price + (atr_val * 7.5), 5)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            side = st.selectbox("Side", ["LONG", "SHORT"], key=f"ord_side_{symbol}")
        with c2:
            risk_pct = st.number_input("Risk %", min_value=0.25, max_value=10.0, value=2.0, step=0.25, key=f"ord_risk_{symbol}")
        with c3:
            sl_price = st.number_input(
                "Stop Loss",
                min_value=0.00001,
                value=float(st.session_state.get(f"ord_sl_{symbol}", default_sl)),
                format="%.5f",
                key=f"ord_sl_{symbol}",
            )
        with c4:
            tp_price = st.number_input(
                "Take Profit",
                min_value=0.00001,
                value=float(st.session_state.get(f"ord_tp_{symbol}", default_tp)),
                format="%.5f",
                key=f"ord_tp_{symbol}",
            )

        stop_dist = abs(latest_price - sl_price)
        reward_dist = abs(tp_price - latest_price)
        rr_preview = (reward_dist / stop_dist) if stop_dist > 0 else 0.0
        risk_dollars = va.balance * (risk_pct / 100.0)

        st.caption(
            f"Entry: `{latest_price:,.5f}` | Risk Amount: `${risk_dollars:,.2f}` ({risk_pct:.2f}%) | "
            f"Projected R:R: `1:{rr_preview:.2f}`"
        )

        if st.button("Execute Virtual Order (With SL/TP)", use_container_width=True):
            ok, msg = va.open_position_with_risk(
                symbol=symbol,
                side=side,
                entry_price=latest_price,
                stop_loss=float(sl_price),
                take_profit=float(tp_price),
                risk_percent=float(risk_pct),
            )
            (st.success if ok else st.error)(msg)
            st.rerun()

        if va.positions:
            st.markdown("**Open Virtual Positions (Auto-Monitored for TP/SL)**")
            for idx, pos in enumerate(va.positions):
                p_col, b_col = st.columns([4.5, 1])
                with p_col:
                    st.write(
                        f"#{idx + 1} | **{pos.symbol}** ({pos.side}) — Entry `{pos.entry_price:,.5f}` | "
                        f"SL `{pos.stop_loss:,.5f}` | TP `{pos.take_profit:,.5f}` | Risk `${pos.risk_usd:,.2f}`"
                    )
                with b_col:
                    if st.button("Close", key=f"close_pos_{current_user}_{idx}", use_container_width=True):
                        ok, msg = va.close_position(idx, latest_price, reason="Manual Close")
                        (st.success if ok else st.error)(msg)
                        st.rerun()

        if va.trade_history:
            st.markdown("**Trade Journal (Closed Trades)**")
            st.dataframe(pd.DataFrame(va.trade_history), use_container_width=True, height=200)

        if st.button("Reset Virtual Balance ($10,000)"):
            va.reset(10_000.0)
            st.rerun()

    else:
        if not bs.connected:
            st.info(
                "Recommended API Brokers: **OANDA v20** or **Capital.com** (Forex/Gold/Oil REST API) "
                "and **Bybit Testnet/Live** (`pybit`/`ccxt` for Crypto). Credentials stay in session memory only."
            )
            with st.form("broker_login_form", clear_on_submit=False):
                b_name = st.selectbox(
                    "Broker / Exchange",
                    ["OANDA (v20 REST)", "Capital.com (REST API)", "Bybit (CCXT / Pybit)", "Exness / MT5 Bridge"],
                )
                b_env = st.selectbox("Environment", ["Testnet / Demo", "Live"])
                b_label = st.text_input("Account Label (Optional)", placeholder="e.g., Destiny Scalp Account")
                b_key = st.text_input("API Key / Token", type="password")
                b_secret = st.text_input("API Secret / Account ID", type="password")
                submitted = st.form_submit_button("Connect Broker Session")

                if submitted:
                    ok, msg = bs.connect(b_name, b_key, b_secret, b_env, b_label)
                    if ok:
                        st.session_state.notifier.push(
                            event_id=f"broker_{current_user}_{b_name}_{b_env}_{pd.Timestamp.utcnow().strftime('%Y%m%d%H')}",
                            category="SECURITY",
                            title="Broker Session Connected",
                            detail=f"{b_label or b_name} ({b_env}) connected for {current_user}.",
                        )
                    (st.success if ok else st.error)(msg)
                    if ok:
                        st.rerun()
        else:
            st.success(
                f"Connected: **{bs.account_label}** ({bs.broker_name}) | "
                f"Env: **{bs.environment}** | Key: `{bs.api_key_masked}`"
            )
            if st.button("Disconnect Broker"):
                bs.disconnect()
                st.rerun()


# =========================================================
# CHART DISPLAY (S/R, OB+FVG, BSL/SSL & LIQUIDITY SWEEPS)
# =========================================================

def display_chart(
    data: pd.DataFrame,
    symbol: str,
    timeframe: str,
    sr_zones: list,
    ob_fvg_zones: list,
    liquidity_levels: list,
    signals_df: pd.DataFrame,
) -> None:
    if data is None or data.empty:
        st.warning("No historical candles are available.")
        return

    chart_data = data.tail(500).copy()
    chart_signals = signals_df.reindex(chart_data.index) if not signals_df.empty else pd.DataFrame(index=chart_data.index)

    c1, c2, c3, c4 = st.columns([1.3, 1.2, 1.2, 1.3])
    with c1:
        visible_count = st.slider("Candles on screen", 30, 180, 80, 10, key=f"candles_{symbol}_{timeframe}")
    with c2:
        chart_height = st.select_slider("Chart height", [550, 650, 750], value=650, key=f"height_{symbol}_{timeframe}")
    with c3:
        st.write("")
        show_zones = st.checkbox("S/R & OB/FVG", value=True, key=f"zones_{symbol}_{timeframe}")
    with c4:
        st.write("")
        show_liq = st.checkbox("BSL/SSL & Sweeps", value=True, key=f"liq_{symbol}_{timeframe}")

    visible = chart_data.tail(visible_count)
    x_start = chart_data.index[0]
    x_end = chart_data.index[-1]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.78, 0.22],
    )

    fig.add_trace(
        go.Candlestick(
            x=chart_data.index,
            open=chart_data["Open"],
            high=chart_data["High"],
            low=chart_data["Low"],
            close=chart_data["Close"],
            name="Price",
            increasing_line_color="#20C997",
            decreasing_line_color="#F05B65",
            increasing_fillcolor="#20C997",
            decreasing_fillcolor="#F05B65",
            whiskerwidth=1,
        ),
        row=1,
        col=1,
    )

    volume_colors = [
        "#20C997" if c >= o else "#F05B65"
        for o, c in zip(chart_data["Open"], chart_data["Close"])
    ]
    fig.add_trace(
        go.Bar(x=chart_data.index, y=chart_data["Volume"], marker_color=volume_colors, name="Volume"),
        row=2,
        col=1,
    )

    if show_zones:
        for z in sr_zones[-10:]:
            fill = "rgba(32, 201, 151, 0.10)" if z.zone_type == "support" else "rgba(240, 91, 101, 0.10)"
            fig.add_hrect(y0=z.bottom, y1=z.top, x0=x_start, x1=x_end, fillcolor=fill, line_width=0, row=1, col=1)

        for pz in ob_fvg_zones[-6:]:
            ob_fill = "rgba(8, 153, 129, 0.16)" if pz.direction == 1 else "rgba(242, 54, 69, 0.16)"
            fvg_fill = "rgba(41, 98, 255, 0.16)" if pz.direction == 1 else "rgba(255, 152, 0, 0.16)"
            fig.add_hrect(y0=pz.ob_low, y1=pz.ob_high, x0=x_start, x1=x_end, fillcolor=ob_fill, line_width=1, row=1, col=1)
            fig.add_hrect(y0=pz.fvg_low, y1=pz.fvg_high, x0=x_start, x1=x_end, fillcolor=fvg_fill, line_width=0, row=1, col=1)

    if show_liq:
        for lv in liquidity_levels[-15:]:
            line_col = "#2962FF" if lv.direction == 1 else "#FF9800"
            tag = "BSL" if lv.direction == 1 else "SSL"
            fig.add_hline(
                y=lv.price,
                line_dash="dash",
                line_width=1,
                line_color=line_col,
                annotation_text=tag,
                annotation_position="top right" if lv.direction == 1 else "bottom right",
                annotation_font_color=line_col,
                annotation_font_size=10,
                row=1,
                col=1,
            )

        if "swept_liquidity_count" in chart_signals.columns:
            swept_mask = chart_signals["swept_liquidity_count"].fillna(0) > 0
            swept_bars = chart_data.loc[swept_mask]
            if not swept_bars.empty:
                fig.add_trace(
                    go.Scatter(
                        x=swept_bars.index,
                        y=swept_bars["High"],
                        mode="markers+text",
                        marker=dict(symbol="x", size=10, color="#FFD54F"),
                        text=["SWEEP"] * len(swept_bars),
                        textposition="top center",
                        textfont=dict(color="#FFD54F", size=10),
                        name="Liquidity Sweep",
                    ),
                    row=1,
                    col=1,
                )

    fig.update_layout(
        title=dict(text=f"{symbol} — {timeframe}", x=0.02, xanchor="left"),
        template="plotly_dark",
        paper_bgcolor="#0B0E11",
        plot_bgcolor="#0B0E11",
        height=chart_height,
        autosize=True,
        margin=dict(l=10, r=75, t=55, b=30),
        font=dict(family="Arial", size=13, color="#E6EAF0"),
        hovermode="x unified",
        dragmode="pan",
        showlegend=False,
        xaxis=dict(
            range=[visible.index[0], visible.index[-1]],
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor="rgba(180,190,200,0.10)",
            nticks=6,
        ),
        xaxis2=dict(showgrid=True, gridcolor="rgba(180,190,200,0.10)", nticks=6),
        yaxis=dict(side="right", showgrid=True, gridcolor="rgba(180,190,200,0.10)", autorange=True, fixedrange=False),
        yaxis2=dict(side="right", showgrid=True, gridcolor="rgba(180,190,200,0.10)", fixedrange=False),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"scrollZoom": True, "displayModeBar": True, "displaylogo": False, "responsive": True},
        key=f"chart_{symbol}_{timeframe}",
    )


# =========================================================
# MAIN DASHBOARD
# =========================================================

st.title("DESTINY AI")
st.caption("Public 5-TF Scalp Workbench | S/R & OB+FVG Overlap | Liquidity Sweeps | Protected Accounts & Alerts")
st.divider()

display_auth_section()

st.divider()
c_mkt, c_asset, c_tf, c_rr = st.columns([1.2, 1.3, 1.0, 1.0])
with c_mkt:
    market_type = st.selectbox("Market", list(MARKETS.keys()))
with c_asset:
    asset_name = st.selectbox("Asset", list(MARKETS[market_type].keys()))
with c_tf:
    selected_timeframe = st.selectbox("Chart Timeframe", TIMEFRAMES, index=2)
with c_rr:
    rr_target = st.number_input("Target R:R", min_value=1.5, max_value=15.0, value=5.0, step=0.5)

symbol = MARKETS[market_type][asset_name]

r_col, info_col = st.columns([1, 3])
with r_col:
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
with info_col:
    st.caption(f"Selected: {market_type} | {asset_name} ({symbol})")

with st.spinner(f"Loading 5 timeframes for {asset_name}..."):
    frames_5tf = load_all_5_timeframes(symbol)
    chart_df = frames_5tf[selected_timeframe] if selected_timeframe in frames_5tf else load_timeframe(symbol, selected_timeframe)

if chart_df.empty:
    st.error("Historical data could not be loaded for this selection.")
    st.stop()

notifier: NotificationCenter = st.session_state.notifier
bar_stamp = str(chart_df.index[-1])
c_high = float(chart_df["High"].iloc[-1])
c_low = float(chart_df["Low"].iloc[-1])

# Evaluate TP/SL for logged-in user and push deduplicated alert
if st.session_state.current_user:
    va_live: VirtualAccount = get_user_virtual_account(st.session_state.current_user)
    tp_sl_events = va_live.evaluate_tp_sl(symbol=symbol, candle_high=c_high, candle_low=c_low)
    for idx_ev, ev in enumerate(tp_sl_events):
        st.toast(ev)
        notifier.push(
            event_id=f"tpsl_{st.session_state.current_user}_{symbol}_{bar_stamp}_{idx_ev}",
            category="TP_SL_HIT",
            title=f"{symbol} Virtual Position Closed",
            detail=ev,
            telegram_token=st.session_state.tg_token,
            telegram_chat_id=st.session_state.tg_chat_id,
        )

mtf_output = run_closed_candle_5tf_cascade(frames_5tf, rr_target=float(rr_target))

if selected_timeframe in mtf_output["per_tf"]:
    active_tf_pack = mtf_output["per_tf"][selected_timeframe]
    sr_res = active_tf_pack["sr"]
    ob_fvg_res = active_tf_pack["ob_fvg"]
else:
    sr_res = run_sr_zones_engine(chart_df)
    ob_fvg_res = run_ob_fvg_engine(chart_df, timeframe_minutes=TF_MINUTES.get(selected_timeframe))

# Push deduplicated alerts for scalp signals and liquidity sweeps
sig = mtf_output.get("scalp_signal")
if sig is not None:
    notifier.push(
        event_id=f"scalp_{symbol}_{sig.direction}_{sig.entry_price}_{bar_stamp}",
        category="SCALP_SIGNAL",
        title=f"{symbol} {sig.direction} ({sig.confidence})",
        detail=f"Entry {sig.entry_price:,.5f} | SL {sig.stop_loss:,.5f} | TP {sig.take_profit:,.5f} (1:{sig.rr_ratio:.1f})",
        telegram_token=st.session_state.tg_token,
        telegram_chat_id=st.session_state.tg_chat_id,
    )

sig_df = ob_fvg_res.get("signals_df", pd.DataFrame())
if not sig_df.empty and int(sig_df["swept_liquidity_count"].iloc[-1]) > 0:
    notifier.push(
        event_id=f"sweep_{symbol}_{selected_timeframe}_{bar_stamp}",
        category="LIQUIDITY_SWEEP",
        title=f"{symbol} ({selected_timeframe}) Liquidity Sweep",
        detail=f"Swept {int(sig_df['swept_liquidity_count'].iloc[-1])} liquidity level(s) at {bar_stamp}.",
        telegram_token=st.session_state.tg_token,
        telegram_chat_id=st.session_state.tg_chat_id,
    )

st.divider()
display_5tf_cascade(mtf_output, symbol)

st.divider()
latest_px = float(chart_df["Close"].iloc[-1])
approx_atr = float((chart_df["High"] - chart_df["Low"]).tail(14).mean())
display_account_panel(symbol, latest_px, approx_atr)

st.divider()
display_notifications_panel()

st.divider()
st.header("Price, Zones & Liquidity Chart")
display_chart(
    data=chart_df,
    symbol=symbol,
    timeframe=selected_timeframe,
    sr_zones=sr_res.get("active_zones", []),
    ob_fvg_zones=ob_fvg_res.get("active_zones", []),
    liquidity_levels=ob_fvg_res.get("active_liquidity", []),
    signals_df=sig_df,
)
