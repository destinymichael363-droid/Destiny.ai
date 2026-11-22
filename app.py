
from __future__ import annotations

import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# =========================================================
# DESTINY AI — HISTORICAL MARKET VIEWER
# Self-contained app.py
# =========================================================

st.set_page_config(
    page_title="DESTINY AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# MARKET DATABASE
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

TIMEFRAMES = [
    "1D",
    "4H",
    "1H",
    "15M",
    "5M",
    "1M",
]

OHLCV = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]

DATA_SETTINGS = {
    "1D": ("1d", "2y"),
    "1H": ("1h", "730d"),
    "15M": ("15m", "60d"),
    "5M": ("5m", "60d"),
    "1M": ("1m", "7d"),
}


# =========================================================
# DATA CLEANING
# =========================================================

def empty_data():
    return pd.DataFrame(columns=OHLCV)


def clean_market_data(frame):

    if frame is None or frame.empty:
        return empty_data()

    frame = frame.copy()

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    required = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    if not all(column in frame.columns for column in required):
        return empty_data()

    if "Volume" not in frame.columns:
        frame["Volume"] = 0

    frame = frame[OHLCV].copy()

    for column in OHLCV:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    frame = frame.sort_index()

    frame = frame[
        ~frame.index.duplicated(keep="last")
    ]

    valid = (
        (frame["High"] >= frame["Open"])
        & (frame["High"] >= frame["Close"])
        & (frame["High"] >= frame["Low"])
        & (frame["Low"] <= frame["Open"])
        & (frame["Low"] <= frame["Close"])
        & (frame["Low"] <= frame["High"])
    )

    return frame.loc[valid]


# =========================================================
# YAHOO FINANCE DATA
# =========================================================

@st.cache_data(
    ttl=300,
    show_spinner=False,
)
def fetch_market_data(symbol, interval, period):

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


# =========================================================
# FOUR-HOUR DATA
# =========================================================

def create_four_hour_data(hourly):

    if hourly is None or hourly.empty:
        return empty_data()

    try:

        result = hourly.resample(
            "4h",
            label="right",
            closed="right",
        ).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })

        return clean_market_data(result)

    except Exception:
        return empty_data()


# =========================================================
# LOAD SELECTED TIMEFRAME
# =========================================================

def load_timeframe(symbol, timeframe):

    if timeframe == "4H":

        hourly = fetch_market_data(
            symbol,
            "1h",
            "730d",
        )

        return create_four_hour_data(hourly)

    interval, period = DATA_SETTINGS[timeframe]

    return fetch_market_data(
        symbol,
        interval,
        period,
    )


# =========================================================
# CHART DISPLAY
# =========================================================

def display_chart(data, symbol, timeframe):

    if data is None or data.empty:
        st.warning("No historical candles are available.")
        return

    chart_data = data.tail(1200).copy()

    first, second = st.columns(2)

    with first:
        visible_count = st.slider(
            "Candles on screen",
            min_value=30,
            max_value=180,
            value=80,
            step=10,
            key=f"candles_{symbol}_{timeframe}",
        )

    with second:
        chart_height = st.select_slider(
            "Chart height",
            options=[550, 650, 750, 850],
            value=650,
            key=f"height_{symbol}_{timeframe}",
        )

    visible = chart_data.tail(visible_count)

    latest_close = float(
        chart_data["Close"].iloc[-1]
    )

    previous_close = (
        float(chart_data["Close"].iloc[-2])
        if len(chart_data) > 1
        else latest_close
    )

    difference = latest_close - previous_close

    change_percent = (
        difference / previous_close * 100
        if previous_close != 0
        else 0
    )

    # -----------------------------------------------------
    # PRICE INFORMATION
    # -----------------------------------------------------

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Latest Close",
        f"{latest_close:,.5f}",
    )

    metric2.metric(
        "Previous Candle Close",
        f"{previous_close:,.5f}",
    )

    metric3.metric(
        "Close-to-Close Change",
        f"{change_percent:+.3f}%",
    )

    # -----------------------------------------------------
    # CHART
    # -----------------------------------------------------

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
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Open: %{open:,.5f}<br>"
                "High: %{high:,.5f}<br>"
                "Low: %{low:,.5f}<br>"
                "Close: %{close:,.5f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    volume_colors = [
        "#20C997" if close >= open else "#F05B65"
        for open, close in zip(
            chart_data["Open"],
            chart_data["Close"],
        )
    ]

    fig.add_trace(
        go.Bar(
            x=chart_data.index,
            y=chart_data["Volume"],
            marker_color=volume_colors,
            name="Volume",
            hovertemplate=(
                "Volume: %{y:,.0f}"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=dict(
            text=f"{symbol} — {timeframe}",
            x=0.02,
            xanchor="left",
        ),
        template="plotly_dark",
        paper_bgcolor="#0B0E11",
        plot_bgcolor="#0B0E11",
        height=chart_height,
        autosize=True,
        margin=dict(
            l=10,
            r=75,
            t=55,
            b=30,
        ),
        font=dict(
            family="Arial",
            size=13,
            color="#E6EAF0",
        ),
        hovermode="x unified",
        dragmode="pan",
        showlegend=False,
        xaxis=dict(
            range=[
                visible.index[0],
                visible.index[-1],
            ],
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor="rgba(180,190,200,0.10)",
            nticks=6,
        ),
        xaxis2=dict(
            showgrid=True,
            gridcolor="rgba(180,190,200,0.10)",
            nticks=6,
        ),
        yaxis=dict(
            side="right",
            showgrid=True,
            gridcolor="rgba(180,190,200,0.10)",
            autorange=True,
            fixedrange=False,
        ),
        yaxis2=dict(
            side="right",
            showgrid=True,
            gridcolor="rgba(180,190,200,0.10)",
            fixedrange=False,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "displaylogo": False,
            "responsive": True,
            "doubleClick": "reset",
            "modeBarButtonsToAdd": [
                "pan2d",
                "zoom2d",
                "zoomIn2d",
                "zoomOut2d",
                "autoScale2d",
                "resetScale2d",
            ],
            "modeBarButtonsToRemove": [
                "select2d",
                "lasso2d",
            ],
        },
        key=f"chart_{symbol}_{timeframe}",
    )

    st.caption(
        "Green candles indicate a close above the open. "
        "Red candles indicate a close below the open. "
        "Use the chart controls to zoom and pan."
    )


# =========================================================
# HISTORICAL DATA TABLE
# =========================================================

def display_data_table(data, timeframe):

    st.subheader("Historical Candle Records")

    if data is None or data.empty:
        st.info("No records available.")
        return

    st.caption(
        f"{timeframe}: {len(data):,} historical candles"
    )

    display = data.tail(100).sort_index(
        ascending=False
    )

    st.dataframe(
        display,
        use_container_width=True,
        height=420,
    )


# =========================================================
# MAIN DASHBOARD
# =========================================================

st.title("DESTINY AI")

st.caption(
    "Historical market data | Candlestick charts | "
    "Multi-market viewing"
)

st.divider()

# ---------------------------------------------------------
# MARKET SELECTION
# ---------------------------------------------------------

market_col, asset_col, timeframe_col = st.columns(3)

with market_col:

    market_type = st.selectbox(
        "Market",
        list(MARKETS.keys()),
    )

with asset_col:

    asset_name = st.selectbox(
        "Asset",
        list(MARKETS[market_type].keys()),
    )

with timeframe_col:

    selected_timeframe = st.selectbox(
        "Timeframe",
        TIMEFRAMES,
        index=2,
    )

symbol = MARKETS[market_type][asset_name]

st.caption(
    f"Selected: {market_type} | "
    f"{asset_name} | {symbol}"
)

# ---------------------------------------------------------
# REFRESH
# ---------------------------------------------------------

refresh_col, information_col = st.columns([1, 3])

with refresh_col:

    if st.button(
        "Refresh Data",
        use_container_width=True,
    ):

        st.cache_data.clear()
        st.rerun()

with information_col:

    st.caption(
        "Historical data provider: Yahoo Finance. "
        "Coverage and availability vary by instrument."
    )

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

with st.spinner(
    f"Loading {asset_name} historical data..."
):

    data = load_timeframe(
        symbol,
        selected_timeframe,
    )

if data.empty:

    st.error(
        "Historical data could not be loaded for this selection. "
        "Try another timeframe or refresh the data."
    )

    st.stop()

# ---------------------------------------------------------
# DATA INFORMATION
# ---------------------------------------------------------

st.divider()
st.header("Market Overview")

info1, info2, info3 = st.columns(3)

info1.metric(
    "Historical Candles",
    f"{len(data):,}",
)

info2.metric(
    "First Available Record",
    str(data.index[0]),
)

info3.metric(
    "Latest Available Record",
    str(data.index[-1]),
)

# ---------------------------------------------------------
# MAIN CHART
# ---------------------------------------------------------

st.divider()
st.header("Historical Price Chart")

display_chart(
    data,
    symbol,
    selected_timeframe,
)

# ---------------------------------------------------------
# HISTORICAL TABLE
# ---------------------------------------------------------

st.divider()

display_data_table(
    data,
    selected_timeframe,
)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "DESTINY AI | Historical market viewer"
)

st.caption(
    "This application displays historical market information. "
    "Data may contain delays, gaps, or provider limitations. "
    "It does not provide trading instructions."
)
