
# =========================================================
# DESTINY AI — HISTORICAL REPLAY VISUALIZER V2
# 15M MINOR S/R + 1H MAJOR S/R
# ONE-CANDLE PREVIOUS / NEXT CONTROLS
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

st.title("Destiny AI — Historical Replay V2")
st.caption(
    "15M minor S/R • 1H major S/R • Historical candle replay"
)


# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close"]

ZONE_COLUMNS = [
    "Side",
    "Zone Low",
    "Zone High",
    "Confirmed Time",
    "Pivot Time",
    "Rejections",
    "Strength",
    "Midline",
    "Strong",
]


# ---------------------------------------------------------
# SYNTHETIC 15-MINUTE DATA
# ---------------------------------------------------------

def create_demo_data(length=480):

    rng = np.random.default_rng(42)

    changes = rng.normal(0, 0.30, length)
    close = 100 + np.cumsum(changes)

    previous = np.concatenate(
        [[close[0]], close[:-1]]
    )

    open_price = previous
    wick = rng.uniform(0.04, 0.20, length)

    high = np.maximum(open_price, close) + wick
    low = np.minimum(open_price, close) - wick

    index = pd.date_range(
        "2025-01-01",
        periods=length,
        freq="15min"
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
# CSV PREPARATION
# ---------------------------------------------------------

def prepare_data(data):

    data = data.copy()

    data.columns = [
        str(column).strip().lower()
        for column in data.columns
    ]

    required = ["open", "high", "low", "close"]

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing OHLC columns: {missing}"
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

    if time_column:

        data[time_column] = pd.to_datetime(
            data[time_column],
            errors="coerce"
        )

        if data[time_column].isna().any():
            raise ValueError(
                "Invalid timestamps found."
            )

        data = data.set_index(time_column)

    else:

        data.index = pd.date_range(
            "2025-01-01",
            periods=len(data),
            freq="15min"
        )

    for column in required:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna(subset=required)
    data = data.sort_index()

    if data.index.has_duplicates:
        raise ValueError(
            "Duplicate candle timestamps found."
        )

    data = data.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
        }
    )

    data = data[REQUIRED_COLUMNS].astype(float)

    if len(data) < 30:
        raise ValueError(
            "At least 30 fifteen-minute candles are required."
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
            "Invalid OHLC candle relationships."
        )

    return data


# ---------------------------------------------------------
# 15M TO 1H AGGREGATION
# ---------------------------------------------------------

def aggregate_1h(completed_15m):

    hourly = completed_15m.resample(
        "1h",
        label="left",
        closed="left"
    ).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        }
    )

    hourly = hourly.dropna()

    return hourly


def completed_hourly_data(prefix):

    # CSV timestamps represent the opening time
    # of each 15-minute candle.
    #
    # An hourly candle is complete only after
    # its fourth 15-minute candle has closed.

    replay_time = prefix.index[-1]
    last_15m_close = (
        replay_time + pd.Timedelta(minutes=15)
    )

    hourly = aggregate_1h(prefix)

    hourly = hourly.loc[
        hourly.index + pd.Timedelta(hours=1)
        <= last_15m_close
    ]

    return hourly


# ---------------------------------------------------------
# EMPTY ZONE RESULT
# ---------------------------------------------------------

def empty_zones():

    return pd.DataFrame(
        columns=ZONE_COLUMNS
    )


# ---------------------------------------------------------
# S/R ENGINE
# ---------------------------------------------------------

def calculate_zones(data):

    if len(data) < 30:
        return empty_zones()

    result = analyze_sr(
        data,
        pivot_length=4,
        atr_length=14,
        strength=3,
        invalid_mode="Wick",
        max_zones=40
    )

    return result


# ---------------------------------------------------------
# SELECT DATA SOURCE
# ---------------------------------------------------------

st.subheader("1. Historical data")

source = st.radio(
    "Data source",
    [
        "Synthetic demonstration",
        "Upload 15M historical CSV"
    ],
    horizontal=True
)

data = None

if source == "Synthetic demonstration":

    data = create_demo_data()

    st.info(
        "Synthetic 15-minute demonstration data. "
        "This is not real market data."
    )

else:

    uploaded = st.file_uploader(
        "Upload 15-minute OHLC CSV",
        type=["csv"]
    )

    if uploaded is not None:

        try:

            raw = pd.read_csv(uploaded)
            data = prepare_data(raw)

            st.success(
                f"{len(data)} fifteen-minute candles loaded."
            )

        except Exception as error:

            st.error(str(error))


if data is None:
    st.info("Load historical data to begin replay.")
    st.stop()


# ---------------------------------------------------------
# REPLAY SETTINGS
# ---------------------------------------------------------

st.subheader("2. Replay settings")

if len(data) < 30:
    st.error("Not enough candles.")
    st.stop()


# Reset replay position when the data source changes
data_signature = (
    len(data),
    str(data.index[0]),
    str(data.index[-1]),
)

if st.session_state.get("replay_data_signature") != data_signature:

    st.session_state["replay_data_signature"] = data_signature
    st.session_state["replay_position"] = len(data)


if "replay_position" not in st.session_state:
    st.session_state["replay_position"] = len(data)

st.session_state["replay_position"] = max(
    30,
    min(
        int(st.session_state["replay_position"]),
        len(data)
    )
)


# ---------------------------------------------------------
# PREVIOUS / NEXT CONTROLS
# ---------------------------------------------------------

st.subheader("3. Candle-by-candle navigation")

previous_col, position_col, next_col = st.columns(
    [1, 1, 1]
)

with previous_col:

    if st.button(
        "← Previous candle",
        use_container_width=True,
        disabled=st.session_state["replay_position"] <= 30
    ):

        st.session_state["replay_position"] = max(
            30,
            st.session_state["replay_position"] - 1
        )

        st.rerun()


with next_col:

    if st.button(
        "Next candle →",
        use_container_width=True,
        disabled=st.session_state["replay_position"] >= len(data)
    ):

        st.session_state["replay_position"] = min(
            len(data),
            st.session_state["replay_position"] + 1
        )

        st.rerun()


with position_col:

    st.write(
        f"**Candle {st.session_state['replay_position']} "
        f"/ {len(data)}**"
    )


selected_candle = st.slider(
    "Replay position",
    min_value=30,
    max_value=len(data),
    key="replay_position"
)


# ---------------------------------------------------------
# HISTORICAL PREFIX
# ---------------------------------------------------------

prefix = data.iloc[:selected_candle].copy()

current_time = prefix.index[-1]
current_close = float(prefix["Close"].iloc[-1])


# ---------------------------------------------------------
# CALCULATE MINOR AND MAJOR S/R
# ---------------------------------------------------------

try:

    # MINOR: 15-minute S/R
    minor_zones = calculate_zones(prefix)

    # MAJOR: completed 1-hour S/R
    hourly_prefix = completed_hourly_data(prefix)

    major_zones = calculate_zones(hourly_prefix)

except Exception as error:

    st.error(
        f"S/R calculation failed: {error}"
    )

    st.stop()


# ---------------------------------------------------------
# REPLAY METRICS
# ---------------------------------------------------------

st.subheader("4. Historical market structure")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "15M candles processed",
        selected_candle
    )

with c2:
    st.metric(
        "Completed 1H candles",
        len(hourly_prefix)
    )

with c3:
    st.metric(
        "Current close",
        f"{current_close:.5f}"
    )

st.write("Replay time:", str(current_time))


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
        name="15M candles",
        increasing_line_color="#16A34A",
        decreasing_line_color="#DC2626",
        increasing_fillcolor="#16A34A",
        decreasing_fillcolor="#DC2626",
    )
)


# ---------------------------------------------------------
# DRAW ZONES
# ---------------------------------------------------------

def draw_zones(
    chart,
    zones,
    timeframe,
    current_time
):

    if zones.empty:
        return

    for _, zone in zones.iterrows():

        side = zone["Side"]

        low = float(zone["Zone Low"])
        high = float(zone["Zone High"])
        midline = float(zone["Midline"])

        confirmed_time = pd.to_datetime(
            zone["Confirmed Time"]
        )

        if confirmed_time > current_time:
            continue

        if timeframe == "1H":

            if side == "Resistance":
                color = "#EF4444"
                fill = "rgba(239,68,68,0.12)"
                label = "1H MAJOR RESISTANCE"
            else:
                color = "#22C55E"
                fill = "rgba(34,197,94,0.12)"
                label = "1H MAJOR SUPPORT"

            dash = "solid"

        else:

            if side == "Resistance":
                color = "#F59E0B"
                fill = "rgba(245,158,11,0.10)"
                label = "15M MINOR RESISTANCE"
            else:
                color = "#38BDF8"
                fill = "rgba(56,189,248,0.10)"
                label = "15M MINOR SUPPORT"

            dash = "dash"

        chart.add_shape(
            type="rect",
            x0=confirmed_time,
            x1=current_time,
            y0=low,
            y1=high,
            fillcolor=fill,
            line=dict(
                color=color,
                width=1.5,
                dash=dash
            ),
            layer="below"
        )

        chart.add_shape(
            type="line",
            x0=confirmed_time,
            x1=current_time,
            y0=midline,
            y1=midline,
            line=dict(
                color=color,
                width=1,
                dash="dot"
            )
        )

        chart.add_annotation(
            x=current_time,
            y=high,
            text=label,
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(
                size=9,
                color=color
            ),
            bgcolor="rgba(15,23,42,0.75)"
        )


# Draw major first, minor second
draw_zones(
    fig,
    major_zones,
    "1H",
    current_time
)

draw_zones(
    fig,
    minor_zones,
    "15M",
    current_time
)


# ---------------------------------------------------------
# CURRENT CANDLE MARKER
# ---------------------------------------------------------

fig.add_vline(
    x=current_time,
    line_width=1,
    line_dash="dot",
    line_color="#94A3B8"
)


# ---------------------------------------------------------
# CHART LAYOUT
# ---------------------------------------------------------

fig.update_layout(
    title="Destiny AI — 15M Historical Replay",
    height=580,
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
# MAJOR S/R TABLE
# ---------------------------------------------------------

st.subheader("5. Major S/R — 1H")

if major_zones.empty:

    st.info(
        "Not enough completed hourly candles to display "
        "major S/R zones yet."
    )

else:

    st.dataframe(
        major_zones[
            [
                "Side",
                "Zone Low",
                "Zone High",
                "Confirmed Time",
                "Pivot Time",
                "Rejections",
                "Midline",
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------
# MINOR S/R TABLE
# ---------------------------------------------------------

st.subheader("6. Minor S/R — 15M")

if minor_zones.empty:

    st.info(
        "No active confirmed 15-minute zones at this point."
    )

else:

    st.dataframe(
        minor_zones[
            [
                "Side",
                "Zone Low",
                "Zone High",
                "Confirmed Time",
                "Pivot Time",
                "Rejections",
                "Midline",
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------
# FINAL NOTICE
# ---------------------------------------------------------

st.divider()

st.caption(
    "Major S/R is calculated from completed 1H candles "
    "aggregated from the available 15M history. "
    "Minor S/R is calculated from the available 15M candles. "
    "Future candles are not passed into either calculation. "
    "CSV timestamps must represent 15M candle opening times."
)
