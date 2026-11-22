
# =========================================================
# DESTINY AI — HISTORICAL REPLAY VISUALIZER V1
# Candle-by-candle S/R replay
# Synthetic or uploaded historical OHLC data
# No live-market signals
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sr_engine import analyze_sr


st.set_page_config(
    page_title="Destiny AI — Historical Replay",
    layout="wide"
)


# ---------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------

st.title("Destiny AI — Historical Replay")
st.caption(
    "Historical candle replay • Confirmed S/R zones only "
    "• No live-market signals"
)


# ---------------------------------------------------------
# SYNTHETIC DEMONSTRATION DATA
# ---------------------------------------------------------

def create_demo_data(length=240):

    rng = np.random.default_rng(42)

    changes = rng.normal(0, 0.55, length)
    close = 100 + np.cumsum(changes)

    previous = np.concatenate(
        [[close[0]], close[:-1]]
    )

    open_price = previous

    wick = rng.uniform(0.08, 0.35, length)

    high = np.maximum(
        open_price,
        close
    ) + wick

    low = np.minimum(
        open_price,
        close
    ) - wick

    index = pd.date_range(
        "2025-01-01",
        periods=length,
        freq="1h"
    )

    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
        },
        index=index
    )


# ---------------------------------------------------------
# CSV DATA PREPARATION
# ---------------------------------------------------------

def prepare_data(data):

    data = data.copy()

    data.columns = [
        str(column).strip().lower()
        for column in data.columns
    ]

    required = [
        "open",
        "high",
        "low",
        "close"
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required CSV columns: {missing}"
        )

    time_column = None

    for candidate in [
        "datetime",
        "timestamp",
        "date",
        "time"
    ]:
        if candidate in data.columns:
            time_column = candidate
            break

    if time_column is not None:

        data[time_column] = pd.to_datetime(
            data[time_column],
            errors="coerce"
        )

        if data[time_column].isna().any():
            raise ValueError(
                "Some timestamps could not be read."
            )

        data = data.set_index(time_column)

    else:
        data.index = pd.date_range(
            "2025-01-01",
            periods=len(data),
            freq="1h"
        )

    for column in required:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna(
        subset=required
    )

    data = data.sort_index()

    data = data.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close"
        }
    )

    if len(data) < 30:
        raise ValueError(
            "At least 30 valid candles are required."
        )

    if not (
        (data["High"] >= data["Low"]).all()
        and
        (data["High"] >= data["Open"]).all()
        and
        (data["High"] >= data["Close"]).all()
        and
        (data["Low"] <= data["Open"]).all()
        and
        (data["Low"] <= data["Close"]).all()
    ):
        raise ValueError(
            "Invalid OHLC candle relationships detected."
        )

    return data[required].astype(float)


# ---------------------------------------------------------
# DATA SOURCE
# ---------------------------------------------------------

st.subheader("1. Historical data")

data_source = st.radio(
    "Choose your data source",
    [
        "Synthetic demonstration",
        "Upload historical CSV"
    ],
    horizontal=True
)

data = None

if data_source == "Synthetic demonstration":

    data = create_demo_data()

    st.info(
        "Using generated demonstration candles. "
        "This is not real market data."
    )

else:

    uploaded_file = st.file_uploader(
        "Upload an OHLC CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            raw_data = pd.read_csv(uploaded_file)
            data = prepare_data(raw_data)

            st.success(
                f"Loaded {len(data)} historical candles."
            )

        except Exception as error:
            st.error(f"Data error: {error}")


if data is None:

    st.info(
        "Choose the demonstration dataset or upload "
        "a valid historical CSV to continue."
    )

    st.stop()


# ---------------------------------------------------------
# S/R ENGINE SETTINGS
# ---------------------------------------------------------

st.subheader("2. S/R engine configuration")

col1, col2, col3 = st.columns(3)

with col1:
    pivot_length = st.number_input(
        "Pivot length",
        min_value=2,
        max_value=30,
        value=4,
        step=1
    )

with col2:
    atr_length = st.number_input(
        "ATR length",
        min_value=2,
        max_value=100,
        value=14,
        step=1
    )

with col3:
    invalid_mode = st.selectbox(
        "Zone invalidation",
        ["Wick", "Close"]
    )


# ---------------------------------------------------------
# REPLAY CONTROL
# ---------------------------------------------------------

st.subheader("3. Candle replay")

if len(data) < 30:
    st.error("Not enough candles for replay.")
    st.stop()

selected_candle = st.slider(
    "Move through historical candles",
    min_value=30,
    max_value=len(data),
    value=len(data),
    step=1
)

# CRITICAL:
# The engine receives only the selected historical prefix.
# No later candles are passed into analyze_sr.

prefix = data.iloc[:selected_candle].copy()

current_time = prefix.index[-1]
current_close = float(prefix["Close"].iloc[-1])


# ---------------------------------------------------------
# REPLAY ENGINE
# ---------------------------------------------------------

try:

    zones = analyze_sr(
        prefix,
        pivot_length=int(pivot_length),
        atr_length=int(atr_length),
        strength=3,
        invalid_mode=invalid_mode,
        max_zones=40
    )

except Exception as error:

    st.error(
        f"S/R engine could not process this replay: {error}"
    )

    st.stop()


# ---------------------------------------------------------
# REPLAY INFORMATION
# ---------------------------------------------------------

st.subheader("4. Historical state")

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "Candles processed",
        f"{selected_candle} / {len(data)}"
    )

with metric2:
    st.metric(
        "Active S/R zones",
        len(zones)
    )

with metric3:
    st.metric(
        "Current candle close",
        f"{current_close:.5f}"
    )

st.write("Replay timestamp:", str(current_time))


# ---------------------------------------------------------
# CANDLESTICK CHART
# ---------------------------------------------------------

fig = go.Figure()

fig.add_trace(
    go.Candlestick(
        x=prefix.index,
        open=prefix["Open"],
        high=prefix["High"],
        low=prefix["Low"],
        close=prefix["Close"],
        name="Historical candles",
        increasing_line_color="#16A34A",
        decreasing_line_color="#DC2626",
        increasing_fillcolor="#16A34A",
        decreasing_fillcolor="#DC2626"
    )
)


# ---------------------------------------------------------
# DRAW CONFIRMED ACTIVE ZONES
# ---------------------------------------------------------

if not zones.empty:

    for _, zone in zones.iterrows():

        side = zone["Side"]

        zone_low = float(zone["Zone Low"])
        zone_high = float(zone["Zone High"])
        midline = float(zone["Midline"])

        confirmed_time = pd.to_datetime(
            zone["Confirmed Time"]
        )

        if side == "Resistance":
            fill = "rgba(220,38,38,0.13)"
            line_color = "#DC2626"
        else:
            fill = "rgba(22,163,74,0.13)"
            line_color = "#16A34A"

        fig.add_shape(
            type="rect",
            x0=confirmed_time,
            x1=current_time,
            y0=zone_low,
            y1=zone_high,
            fillcolor=fill,
            line=dict(
                color=line_color,
                width=1
            ),
            layer="below"
        )

        fig.add_shape(
            type="line",
            x0=confirmed_time,
            x1=current_time,
            y0=midline,
            y1=midline,
            line=dict(
                color=line_color,
                width=1,
                dash="dash"
            )
        )


# ---------------------------------------------------------
# CURRENT CANDLE MARKER
# ---------------------------------------------------------

fig.add_vline(
    x=current_time,
    line_width=1,
    line_dash="dot",
    line_color="#64748B"
)


fig.update_layout(
    title="Destiny AI — Historical S/R Replay",
    height=560,
    template="plotly_dark",
    xaxis_title="Historical time",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
    margin=dict(
        l=45,
        r=25,
        t=65,
        b=45
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

fig.update_xaxes(
    showgrid=True,
    rangeslider_visible=False
)

fig.update_yaxes(
    showgrid=True,
    fixedrange=False
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ---------------------------------------------------------
# ACTIVE ZONE TABLE
# ---------------------------------------------------------

st.subheader("5. Active confirmed S/R zones")

if zones.empty:

    st.warning(
        "No active confirmed zones at this replay point."
    )

else:

    display_columns = [
        "Side",
        "Zone Low",
        "Zone High",
        "Confirmed Time",
        "Pivot Time",
        "Rejections",
        "Strength",
        "Midline",
        "Strong"
    ]

    st.dataframe(
        zones[display_columns],
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------
# HISTORICAL LIMITATION NOTICE
# ---------------------------------------------------------

st.divider()

st.caption(
    "Replay integrity: the S/R engine receives only the "
    "historical candles up to the selected replay point. "
    "Zones are recalculated at each selection. "
    "This visualization does not establish profitability "
    "or guarantee complete absence of look-ahead bias."
)

