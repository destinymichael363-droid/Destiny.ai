from __future__ import annotations

from datetime import datetime, timezone
import urllib.request

import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

from auth import get_authenticated_supabase
from auth_guard import require_login
from paper_engine import load_account
from mtf_context import build_mtf_context
from strategy_engine import analyze_market
from sr_engine import analyze_sr


# =========================================================
# DESTINY AI — HISTORICAL ANALYSIS / EDUCATIONAL REVIEW
# =========================================================

st.set_page_config(
    page_title="Destiny AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

require_login()
supabase = get_authenticated_supabase()

if supabase is None:
    st.warning("Please log in again.")
    st.stop()

user_id = st.session_state["supabase_user"]

try:
    account = load_account(supabase, user_id)
    st.metric(
        "Virtual Practice Balance",
        f"{account['balance']:,.2f}"
    )
except Exception as exc:
    st.error(f"Could not load virtual account: {exc}")


TIMEFRAMES = ["1D", "4H", "1H", "15M", "5M", "1M"]

SR_TIMEFRAMES = ["4H", "1H", "15M"]

SR_CLASSIFICATION = {
    "4H": "MAJOR",
    "1H": "MAJOR",
    "15M": "MINOR",
}

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

OHLCV_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


# =========================================================
# OPTIONAL NOTIFICATION
# =========================================================

def send_app_notification(title, message):

    try:
        topic = str(
            st.secrets.get("NTFY_TOPIC", "")
        ).strip()

        if not topic:
            return False

        request = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=str(message).encode("utf-8"),
            headers={"Title": str(title)},
            method="POST",
        )

        with urllib.request.urlopen(
            request,
            timeout=10
        ) as response:

            return response.status in (200, 201)

    except Exception:
        return False


if st.sidebar.button(
    "Test Android Notification"
):

    if send_app_notification(
        "Destiny AI",
        "Destiny AI notification test completed."
    ):
        st.sidebar.success("Test notification sent.")
    else:
        st.sidebar.warning(
            "Notification unavailable. Check NTFY_TOPIC."
        )


# =========================================================
# HISTORICAL MARKET DATA
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_data(symbol, interval, period):

    try:

        frame = yf.Ticker(symbol).history(
            period=period,
            interval=interval,
            auto_adjust=False,
            actions=False,
            prepost=False,
        )

        if frame is None or frame.empty:
            return pd.DataFrame(
                columns=OHLCV_COLUMNS
            )

        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = (
                frame.columns.get_level_values(0)
            )

        required = [
            "Open",
            "High",
            "Low",
            "Close",
        ]

        if not all(
            column in frame.columns
            for column in required
        ):
            return pd.DataFrame(
                columns=OHLCV_COLUMNS
            )

        if "Volume" not in frame.columns:
            frame["Volume"] = 0

        frame = frame[OHLCV_COLUMNS].copy()

        for column in OHLCV_COLUMNS:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce"
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
            frame["High"]
            >= frame[
                ["Open", "Close", "Low"]
            ].max(axis=1)
        ) & (
            frame["Low"]
            <= frame[
                ["Open", "Close", "High"]
            ].min(axis=1)
        )

        return frame.loc[valid]

    except Exception:
        return pd.DataFrame(
            columns=OHLCV_COLUMNS
        )


def remove_latest_candle(frame):

    if (
        frame is None
        or frame.empty
        or len(frame) < 2
    ):
        return pd.DataFrame(
            columns=OHLCV_COLUMNS
        )

    return frame.iloc[:-1].copy()


def build_four_hour_data(hourly):

    if (
        hourly is None
        or hourly.empty
        or len(hourly) < 5
    ):
        return pd.DataFrame(
            columns=OHLCV_COLUMNS
        )

    try:

        resampled = hourly.resample(
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

        resampled = resampled.dropna(
            subset=[
                "Open",
                "High",
                "Low",
                "Close",
            ]
        )

        resampled = resampled[OHLCV_COLUMNS]

        if len(resampled) > 1:
            return resampled.iloc[:-1].copy()

        return pd.DataFrame(
            columns=OHLCV_COLUMNS
        )

    except Exception:
        return pd.DataFrame(
            columns=OHLCV_COLUMNS
        )


@st.cache_data(ttl=300, show_spinner=False)
def load_all_timeframes(symbol):

    daily = remove_latest_candle(
        fetch_market_data(
            symbol,
            "1d",
            "2y"
        )
    )

    hourly = remove_latest_candle(
        fetch_market_data(
            symbol,
            "1h",
            "730d"
        )
    )

    fifteen = remove_latest_candle(
        fetch_market_data(
            symbol,
            "15m",
            "60d"
        )
    )

    five = remove_latest_candle(
        fetch_market_data(
            symbol,
            "5m",
            "60d"
        )
    )

    one = remove_latest_candle(
        fetch_market_data(
            symbol,
            "1m",
            "7d"
        )
    )

    return {
        "1D": daily,
        "4H": build_four_hour_data(hourly),
        "1H": hourly,
        "15M": fifteen,
        "5M": five,
        "1M": one,
}
# =========================================================
# HISTORICAL ANALYSIS ENGINES
# =========================================================

@st.cache_data(ttl=1800, show_spinner=False)
def run_all_analyses(frames):

    results = {}

    for timeframe in TIMEFRAMES:

        data = frames.get(
            timeframe,
            pd.DataFrame()
        )

        if data is None or data.empty:

            results[timeframe] = {
                "status": "UNAVAILABLE",
                "error": "Historical data unavailable.",
            }

            continue

        try:

            output = analyze_market(
                data.tail(1200).copy()
            )

            if not isinstance(output, dict):
                raise TypeError(
                    "Analysis engine returned an unexpected result type."
                )

            results[timeframe] = {
                "status": "ANALYSED",
                "data": output,
                "error": None,
            }

        except Exception as exc:

            results[timeframe] = {
                "status": "ERROR",
                "error": str(exc),
            }

    return results


# =========================================================
# SUPPORT AND RESISTANCE ANALYSIS
# =========================================================

@st.cache_data(ttl=1800, show_spinner=False)
def run_sr_analysis(data, source_timeframe):

    if data is None or data.empty:
        return pd.DataFrame()

    clean = data.copy().sort_index()

    clean = clean[
        ~clean.index.duplicated(keep="last")
    ]

    required = [
        "High",
        "Low",
        "Close",
    ]

    if not all(
        column in clean.columns
        for column in required
    ):

        raise ValueError(
            f"{source_timeframe}: required OHLC columns are missing."
        )

    clean = clean.dropna(
        subset=required
    )

    if len(clean) < 30:
        return pd.DataFrame()

    zones = analyze_sr(
        data=clean,
        pivot_length=8,
        zone_atr=1.0,
        atr_length=14,
        strength=3,
        invalid_mode="Close",
        max_zones=40,
    )

    if not isinstance(zones, pd.DataFrame):

        raise TypeError(
            f"{source_timeframe}: S/R engine returned a non-DataFrame result."
        )

    if zones.empty:
        return pd.DataFrame()

    zones = zones.copy()

    if "Confirmed Time" not in zones.columns:

        raise ValueError(
            f"{source_timeframe}: S/R output has no Confirmed Time column."
        )

    zones["Source Timeframe"] = source_timeframe

    zones["Classification"] = (
        SR_CLASSIFICATION.get(
            source_timeframe,
            "UNCLASSIFIED",
        )
    )

    return (
        zones.sort_values(
            "Confirmed Time",
            ascending=False
        )
        .head(8)
        .reset_index(drop=True)
    )


@st.cache_data(ttl=1800, show_spinner=False)
def build_combined_sr(frames):

    layers = []

    for timeframe in SR_TIMEFRAMES:

        source = frames.get(
            timeframe,
            pd.DataFrame()
        )

        layer = run_sr_analysis(
            source,
            timeframe
        )

        if (
            isinstance(layer, pd.DataFrame)
            and not layer.empty
        ):

            layers.append(layer)

    if layers:

        return pd.concat(
            layers,
            ignore_index=True
        )

    return pd.DataFrame()


# =========================================================
# TIME ALIGNMENT HELPER
# =========================================================

def align_timestamp(value, reference):

    timestamp = pd.Timestamp(value)

    reference_timestamp = pd.Timestamp(
        reference
    )

    if (
        reference_timestamp.tzinfo is None
        and timestamp.tzinfo is not None
    ):

        timestamp = timestamp.tz_localize(None)

    elif (
        reference_timestamp.tzinfo is not None
        and timestamp.tzinfo is None
    ):

        timestamp = timestamp.tz_localize(
            reference_timestamp.tzinfo
        )

    return timestamp
# =========================================================
# SUPPORT AND RESISTANCE CHART DISPLAY
# =========================================================

def draw_sr_zones(
    fig,
    zones,
    data,
    start_time,
    end_time,
):

    required = [
        "Side",
        "Zone Low",
        "Zone High",
    ]

    if (
        not isinstance(zones, pd.DataFrame)
        or zones.empty
        or not all(
            column in zones.columns
            for column in required
        )
    ):
        return

    try:
        latest_close = float(
            data["Close"].iloc[-1]
        )
    except Exception:
        return

    parsed = []

    for _, zone in zones.iterrows():

        try:
            side = str(
                zone["Side"]
            ).upper()

            low = float(
                zone["Zone Low"]
            )

            high = float(
                zone["Zone High"]
            )

            source_tf = str(
                zone.get(
                    "Source Timeframe",
                    "S/R"
                )
            )

            classification = str(
                zone.get(
                    "Classification",
                    "UNCLASSIFIED"
                )
            )

            if (
                side not in (
                    "SUPPORT",
                    "RESISTANCE"
                )
                or pd.isna(low)
                or pd.isna(high)
                or low >= high
            ):
                continue

            distance = abs(
                (low + high) / 2
                - latest_close
            )

            parsed.append(
                (
                    distance,
                    zone,
                    side,
                    low,
                    high,
                    source_tf,
                    classification,
                )
            )

        except Exception:
            continue

    if not parsed:
        return

    # Keep up to three nearest zones
    # from each S/R timeframe.

    by_layer = {}

    for item in sorted(
        parsed,
        key=lambda row: row[0]
    ):

        by_layer.setdefault(
            item[5],
            []
        )

        if len(by_layer[item[5]]) < 3:
            by_layer[item[5]].append(item)

    selected = [
        item
        for layer_items in by_layer.values()
        for item in layer_items
    ]

    selected.sort(
        key=lambda row: row[0]
    )

    label_offset = 0

    for (
        _,
        zone,
        side,
        low,
        high,
        source_tf,
        classification,
    ) in selected:

        try:

            zone_time = zone.get(
                "Confirmed Time",
                zone.get(
                    "Pivot Time",
                    data.index[0]
                )
            )

            if pd.isna(zone_time):
                zone_time = data.index[0]

            zone_start = max(
                align_timestamp(
                    zone_time,
                    start_time
                ),
                start_time,
            )

            if zone_start > end_time:
                continue

            color = (
                "#00C087"
                if side == "SUPPORT"
                else "#F04452"
            )

            dash = {
                "4H": "solid",
                "1H": "dash",
                "15M": "dot",
            }.get(
                source_tf,
                "solid"
            )

            alpha = {
                "4H": 0.105,
                "1H": 0.075,
                "15M": 0.045,
            }.get(
                source_tf,
                0.06
            )

            if side == "SUPPORT":

                fill = (
                    f"rgba(0,192,135,{alpha})"
                )

            else:

                fill = (
                    f"rgba(240,68,82,{alpha})"
                )

            midpoint = (
                low + high
            ) / 2

            fig.add_shape(
                type="rect",
                x0=zone_start,
                x1=end_time,
                y0=low,
                y1=high,
                fillcolor=fill,
                line=dict(
                    color=color,
                    width=(
                        1.25
                        if source_tf == "4H"
                        else 1
                    ),
                    dash=dash,
                ),
                layer="below",
            )

            fig.add_shape(
                type="line",
                x0=zone_start,
                x1=end_time,
                y0=midpoint,
                y1=midpoint,
                line=dict(
                    color=color,
                    width=0.8,
                    dash=dash,
                ),
                layer="below",
            )

            fig.add_annotation(
                x=end_time,
                y=midpoint,
                text=(
                    f"{source_tf} "
                    f"{classification} · {side}"
                ),
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                yshift=(
                    label_offset % 3 - 1
                ) * 12,
                font=dict(
                    size=8,
                    color=color
                ),
                bgcolor="rgba(11,14,17,0.82)",
                borderpad=2,
            )

            label_offset += 1

        except Exception:
            continue


# =========================================================
# ORDER BLOCK + FVG OVERLAP DISPLAY
# =========================================================

def draw_ob_fvg_overlaps(
    fig,
    setups,
    data,
    start_time,
    end_time,
):

    required = [
        "ob_time",
        "ob_low",
        "ob_high",
        "fvg_time",
        "fvg_low",
        "fvg_high",
        "direction",
    ]

    if (
        not isinstance(setups, pd.DataFrame)
        or setups.empty
        or not all(
            column in setups.columns
            for column in required
        )
    ):
        return 0

    drawn = 0
    active_label_count = 0

    for _, setup in setups.tail(100).iterrows():

        if drawn >= 5:
            break

        try:

            ob_time = setup["ob_time"]
            fvg_time = setup["fvg_time"]

            if (
                pd.isna(ob_time)
                or pd.isna(fvg_time)
            ):
                continue

            overlap_time = max(
                align_timestamp(
                    ob_time,
                    start_time
                ),
                align_timestamp(
                    fvg_time,
                    start_time
                ),
            )

            if overlap_time > end_time:
                continue

            ob_low = float(
                setup["ob_low"]
            )

            ob_high = float(
                setup["ob_high"]
            )

            fvg_low = float(
                setup["fvg_low"]
            )

            fvg_high = float(
                setup["fvg_high"]
            )

            # Calculate shared price area.

            low = max(
                ob_low,
                fvg_low
            )

            high = min(
                ob_high,
                fvg_high
            )

            if low >= high:
                continue

            direction = str(
                setup["direction"]
            ).upper()

            if direction == "BULLISH":

                color = "#00C087"

                invalid_mask = (
                    (data.index >= overlap_time)
                    & (
                        data["Close"]
                        < max(
                            ob_low,
                            fvg_low
                        )
                    )
                )

            elif direction == "BEARISH":

                color = "#F04452"

                invalid_mask = (
                    (data.index >= overlap_time)
                    & (
                        data["Close"]
                        > min(
                            ob_high,
                            fvg_high
                        )
                    )
                )

            else:
                continue

            invalid_rows = data.loc[
                invalid_mask
            ]

            setup_end = end_time

            invalidated = (
                not invalid_rows.empty
            )

            if invalidated:

                invalid_time = align_timestamp(
                    invalid_rows.index[0],
                    start_time
                )

                if invalid_time < start_time:
                    continue

                setup_end = min(
                    invalid_time,
                    end_time
                )

            elif (
                str(
                    setup.get(
                        "status",
                        "ACTIVE"
                    )
                ).upper()
                == "INVALIDATED"
            ):
                continue

            visible_start = max(
                overlap_time,
                start_time
            )

            if visible_start > setup_end:
                continue

            if direction == "BEARISH":

                fill = (
                    "rgba(240,68,82,0.035)"
                    if invalidated
                    else "rgba(240,68,82,0.12)"
                )

            else:

                fill = (
                    "rgba(0,192,135,0.035)"
                    if invalidated
                    else "rgba(0,192,135,0.12)"
                )

            fig.add_shape(
                type="rect",
                x0=visible_start,
                x1=setup_end,
                y0=low,
                y1=high,
                fillcolor=fill,
                line=dict(
                    color=color,
                    width=(
                        0.7
                        if invalidated
                        else 1
                    ),
                    dash=(
                        "dot"
                        if invalidated
                        else "solid"
                    ),
                ),
                layer="below",
            )

            if (
                not invalidated
                and active_label_count < 2
            ):

                label = (
                    "BULL OB/FVG"
                    if direction == "BULLISH"
                    else "BEAR OB/FVG"
                )

                fig.add_annotation(
                    x=end_time,
                    y=(low + high) / 2,
                    text=label,
                    showarrow=False,
                    xanchor="right",
                    yanchor="middle",
                    yshift=(
                        active_label_count * 12
                    ),
                    font=dict(
                        size=8,
                        color=color
                    ),
                    bgcolor="rgba(11,14,17,0.82)",
                )

                active_label_count += 1

            drawn += 1

        except Exception:
            continue

    return drawn
# =========================================================
# MOBILE-FRIENDLY HISTORICAL CHART
# =========================================================

def display_chart(
    data,
    result,
    sr_zones,
    symbol,
    timeframe,
):

    chart_data = data.tail(1200).copy()

    if chart_data.empty:
        st.warning("No candles are available.")
        return

    left, right = st.columns(2)

    with left:

        visible_count = st.slider(
            "Candles on screen",
            30,
            180,
            60,
            10,
            key=f"visible_{symbol}_{timeframe}",
        )

    with right:

        vertical_scale = st.slider(
            "Vertical zoom",
            0.6,
            2.0,
            1.0,
            0.1,
            key=f"vertical_{symbol}_{timeframe}",
            help="Adjust the visible vertical price range.",
        )

    start_time = chart_data.index[0]
    end_time = chart_data.index[-1]

    latest_close = float(
        chart_data["Close"].iloc[-1]
    )

    visible = chart_data.tail(
        visible_count
    )

    first_visible = visible.index[0]

    visible_low = float(
        visible["Low"].min()
    )

    visible_high = float(
        visible["High"].max()
    )

    span = max(
        visible_high - visible_low,
        abs(latest_close) * 0.002,
    )

    center = (
        visible_high + visible_low
    ) / 2

    half_range = max(
        span * 0.55 * vertical_scale,
        abs(latest_close) * 0.0005,
    )

    y_range = [
        center - half_range,
        center + half_range,
    ]

    fig = go.Figure()

    volume_custom = (
        chart_data["Volume"]
        .fillna(0)
        .to_numpy()
        .reshape(-1, 1)
    )

    # -----------------------------------------------------
    # PRICE CANDLES
    # -----------------------------------------------------

    fig.add_trace(
        go.Candlestick(
            x=chart_data.index,
            open=chart_data["Open"],
            high=chart_data["High"],
            low=chart_data["Low"],
            close=chart_data["Close"],
            customdata=volume_custom,
            name="Price",

            increasing_line_color="#20C997",
            decreasing_line_color="#F05B65",

            increasing_fillcolor="#20C997",
            decreasing_fillcolor="#F05B65",

            whiskerwidth=1,

            hovertemplate=(
                "<b>%{x|%d %b %Y %H:%M}</b><br>"
                "Open: %{open:,.5f}<br>"
                "High: %{high:,.5f}<br>"
                "Low: %{low:,.5f}<br>"
                "Close: %{close:,.5f}<br>"
                "Volume: %{customdata[0]:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    # -----------------------------------------------------
    # SUPPORT AND RESISTANCE
    # -----------------------------------------------------

    draw_sr_zones(
        fig,
        sr_zones,
        chart_data,
        start_time,
        end_time,
    )

    # -----------------------------------------------------
    # OB / FVG OVERLAPS
    # -----------------------------------------------------

    overlap_count = draw_ob_fvg_overlaps(
        fig,
        result.get(
            "setups",
            pd.DataFrame()
        ),
        chart_data,
        start_time,
        end_time,
    )

    # -----------------------------------------------------
    # BSL / SSL LEVELS
    # -----------------------------------------------------

    liquidity = result.get(
        "liquidity",
        pd.DataFrame()
    )

    if (
        isinstance(liquidity, pd.DataFrame)
        and not liquidity.empty
        and {
            "swept",
            "price",
            "side",
        }.issubset(liquidity.columns)
    ):

        active_liquidity = liquidity[
            liquidity["swept"].eq(False)
        ].tail(6)

        for _, level in active_liquidity.iterrows():

            try:

                price = float(
                    level["price"]
                )

                color = (
                    "#F05B65"
                    if str(
                        level["side"]
                    ).upper() == "BSL"
                    else "#20C997"
                )

                fig.add_hline(
                    y=price,
                    line_dash="dot",
                    line_color=color,
                    line_width=1.2,
                )

            except Exception:
                continue

    # -----------------------------------------------------
    # LATEST COMPLETED CANDLE PRICE
    # -----------------------------------------------------

    fig.add_hline(
        y=latest_close,
        line_color="#FFD54F",
        line_width=1.7,
        annotation_text=(
            f"Latest close {latest_close:,.5f}"
        ),
        annotation_position="top right",
        annotation_font=dict(
            color="#FFD54F",
            size=10,
        ),
        annotation_bgcolor=(
            "rgba(20,20,20,0.88)"
        ),
    )

    reset_counter = st.session_state.get(
        "destiny_chart_reset_counter",
        0,
    )

    # -----------------------------------------------------
    # CHART LAYOUT
    # -----------------------------------------------------

    fig.update_layout(

        title=dict(
            text=f"{symbol} | {timeframe}",
            x=0.02,
            xanchor="left",
            font=dict(
                size=18,
                color="#E6EAF0",
            ),
        ),

        template="plotly_dark",

        paper_bgcolor="#0B0E11",
        plot_bgcolor="#0B0E11",

        height=780,
        autosize=True,

        margin=dict(
            l=8,
            r=82,
            t=58,
            b=35,
        ),

        font=dict(
            family="Arial, sans-serif",
            size=13,
            color="#D7DCE2",
        ),

        hoverlabel=dict(
            bgcolor="#20262D",
            bordercolor="#59636E",
            font=dict(
                size=12,
                color="#FFFFFF",
            ),
        ),

        hovermode="closest",
        dragmode="pan",
        showlegend=False,

        uirevision=(
            f"destiny-{symbol}-"
            f"{timeframe}-{reset_counter}"
        ),

        xaxis=dict(
            type="date",
            range=[
                first_visible,
                end_time,
            ],
            rangeslider=dict(
                visible=False
            ),
            tickformat="%d %b",
            nticks=5,
            showgrid=True,
            gridcolor=(
                "rgba(180,190,200,0.10)"
            ),
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            automargin=True,
        ),

        yaxis=dict(
            side="right",
            range=y_range,
            autorange=False,
            fixedrange=False,
            showgrid=True,
            gridcolor=(
                "rgba(180,190,200,0.10)"
            ),
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            automargin=True,
        ),
    )

    # -----------------------------------------------------
    # DISPLAY CHART
    # -----------------------------------------------------

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

        key=(
            f"destiny-chart-"
            f"{symbol}-{timeframe}"
        ),
    )

    st.caption(
        "4H major S/R: solid | "
        "1H major: dashed | "
        "15M minor: dotted. "
        "Green: support/bullish; "
        "red: resistance/bearish. "
        "Yellow: latest completed close. "
        "S/R invalidation is close-based "
        "on its source timeframe."
    )

    if overlap_count == 0:

        st.info(
            "No qualified OB/FVG overlap is visible "
            "in this chart window. "
            "Standalone FVGs are not drawn."
    )
# =========================================================
# PART 5 — HISTORICAL REVIEW AND STATISTICS
# =========================================================

def build_historical_summary(frames, analyses):
    rows = []

    for timeframe in TIMEFRAMES:
        candles = frames.get(timeframe, pd.DataFrame())
        analysis = analyses.get(timeframe, {})

        if candles is None or candles.empty:
            rows.append({
                "Timeframe": timeframe,
                "Status": "NO DATA",
                "Candles": 0,
                "First Candle": "—",
                "Last Candle": "—",
                "Missing Values": "—",
            })
            continue

        missing = int(
            candles[OHLCV_COLUMNS]
            .isna()
            .sum()
            .sum()
        )

        rows.append({
            "Timeframe": timeframe,
            "Status": analysis.get("status", "NOT ANALYSED"),
            "Candles": len(candles),
            "First Candle": str(candles.index[0]),
            "Last Candle": str(candles.index[-1]),
            "Missing Values": missing,
        })

    return pd.DataFrame(rows)


def display_historical_summary(frames, analyses):
    st.subheader("Historical Data Review")

    summary = build_historical_summary(
        frames,
        analyses,
    )

    if summary.empty:
        st.info("No historical data is available.")
        return

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )

    total_candles = int(summary["Candles"].sum())
    successful = int(
        summary["Status"].eq("ANALYSED").sum()
    )
    unavailable = int(
        summary["Status"].eq("UNAVAILABLE").sum()
    )
    errors = int(
        summary["Status"].eq("ERROR").sum()
    )

    first, second, third, fourth = st.columns(4)

    first.metric(
        "Total Historical Candles",
        f"{total_candles:,}",
    )

    second.metric(
        "Analysed Timeframes",
        f"{successful}/{len(TIMEFRAMES)}",
    )

    third.metric(
        "Unavailable",
        unavailable,
    )

    fourth.metric(
        "Analysis Errors",
        errors,
    )


def display_analysis_status(analyses):
    st.subheader("Timeframe Analysis Status")

    for timeframe in TIMEFRAMES:
        result = analyses.get(timeframe, {})
        status = result.get("status", "UNKNOWN")

        if status == "ANALYSED":
            st.success(
                f"{timeframe}: Historical analysis completed."
            )

        elif status == "UNAVAILABLE":
            st.warning(
                f"{timeframe}: Historical data unavailable."
            )

        else:
            error = result.get(
                "error",
                "Unknown analysis error.",
            )
            st.error(
                f"{timeframe}: {error}"
            )


def display_historical_candle_details(frames):
    st.subheader("Historical Candle Details")

    available = [
        timeframe
        for timeframe in TIMEFRAMES
        if (
            isinstance(
                frames.get(timeframe),
                pd.DataFrame,
            )
            and not frames[timeframe].empty
        )
    ]

    if not available:
        st.info("There are no candles to display.")
        return

    selected = st.selectbox(
        "Select historical timeframe",
        available,
        key="historical_detail_timeframe",
    )

    candles = frames[selected].copy()

    st.caption(
        f"{selected} historical records: {len(candles):,}"
    )

    st.dataframe(
        candles.tail(100).sort_index(
            ascending=False
        ),
        use_container_width=True,
    )


def display_historical_review(frames, analyses):
    st.header("Historical Market Review")

    display_historical_summary(
        frames,
        analyses,
    )

    st.divider()

    display_analysis_status(
        analyses,
    )

    st.divider()

    display_historical_candle_details(
        frames,
    )
# =========================================================
# PART 6 — MAIN DESTINY AI DASHBOARD
# =========================================================

st.title("DESTINY AI")
st.caption(
    "Historical market analysis | Multi-timeframe review | "
    "Virtual practice"
)

# ---------------------------------------------------------
# MARKET SELECTION
# ---------------------------------------------------------

market_col, asset_col, timeframe_col = st.columns(3)

with market_col:
    market_type = st.selectbox(
        "Market",
        list(MARKETS.keys()),
        key="market_type",
    )

with asset_col:
    asset_name = st.selectbox(
        "Asset",
        list(MARKETS[market_type].keys()),
        key=f"asset_{market_type}",
    )

with timeframe_col:
    selected_timeframe = st.selectbox(
        "Chart timeframe",
        TIMEFRAMES,
        index=2,
        key="selected_timeframe",
    )

symbol = MARKETS[market_type][asset_name]

st.caption(
    f"Selected market: {market_type} | "
    f"Asset: {asset_name} | Symbol: {symbol}"
)

# ---------------------------------------------------------
# LOAD HISTORICAL DATA
# ---------------------------------------------------------

refresh_col, status_col = st.columns([1, 3])

with refresh_col:
    if st.button(
        "Refresh Historical Data",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.rerun()

with status_col:
    st.caption(
        "Data is retrieved from Yahoo Finance. "
        "Availability and historical coverage vary by market."
    )

with st.spinner("Loading historical market data..."):
    frames = load_all_timeframes(symbol)

selected_data = frames.get(
    selected_timeframe,
    pd.DataFrame(),
)

if selected_data.empty:
    st.error(
        f"No historical candles are available for "
        f"{asset_name} on {selected_timeframe}."
    )
    st.stop()

# ---------------------------------------------------------
# RUN HISTORICAL ANALYSIS
# ---------------------------------------------------------

with st.spinner("Reviewing historical market structure..."):
    analyses = run_all_analyses(frames)

selected_result = analyses.get(
    selected_timeframe,
    {},
)

if selected_result.get("status") != "ANALYSED":
    st.warning(
        selected_result.get(
            "error",
            "Historical analysis could not be completed.",
        )
    )
    selected_result = {}

result_data = selected_result.get(
    "data",
    {},
)

if not isinstance(result_data, dict):
    result_data = {}

# ---------------------------------------------------------
# SUPPORT AND RESISTANCE
# ---------------------------------------------------------

with st.spinner("Loading historical support and resistance..."):
    sr_zones = build_combined_sr(frames)

st.subheader("Support and Resistance")

if isinstance(sr_zones, pd.DataFrame) and not sr_zones.empty:
    display_columns = [
        column
        for column in [
            "Source Timeframe",
            "Classification",
            "Side",
            "Zone Low",
            "Zone High",
            "Confirmed Time",
        ]
        if column in sr_zones.columns
    ]

    st.dataframe(
        sr_zones[display_columns],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info(
        "No confirmed support or resistance zones "
        "are available in the selected historical data."
    )

# ---------------------------------------------------------
# MAIN PRICE CHART
# ---------------------------------------------------------

st.divider()
st.header("Historical Price Chart")

display_chart(
    selected_data,
    result_data,
    sr_zones,
    asset_name,
    selected_timeframe,
)


# ---------------------------------------------------------
# MULTI-TIMEFRAME CONTEXT — READABLE DISPLAY
# ---------------------------------------------------------

st.divider()
st.header("Multi-Timeframe Context")

try:
    mtf_result = build_mtf_context(analyses)

    if isinstance(mtf_result, dict):
        timeframe_data = mtf_result.get("timeframes", {})

        if timeframe_data:
            mtf_rows = []

            for tf, details in timeframe_data.items():
                mtf_rows.append({
                    "Timeframe": tf,
                    "Status": details.get("status", "Unknown"),
                    "Structure": details.get("structure", "Unknown"),
                    "Bullish BOS": details.get("bullish_bos", 0),
                    "Bearish BOS": details.get("bearish_bos", 0),
                    "Candles": details.get("candle_count", 0),
                })

            mtf_df = pd.DataFrame(mtf_rows)

            st.dataframe(
                mtf_df,
                use_container_width=True,
                hide_index=True,
            )

        st.info(
            mtf_result.get(
                "higher_timeframe_summary",
                "Higher timeframe summary unavailable."
            )
        )

        st.caption(
            mtf_result.get(
                "lower_timeframe_summary",
                "Lower timeframe summary unavailable."
            )
        )

        st.caption(
            "Data coverage: "
            + str(
                mtf_result.get(
                    "data_coverage",
                    "Unknown"
                )
            )
        )

    else:
        st.warning("Multi-timeframe data is unavailable.")

except Exception as exc:
    st.warning(
        f"Multi-timeframe context unavailable: {exc}"
    )


# ---------------------------------------------------------
# HISTORICAL REVIEW
# ---------------------------------------------------------

st.divider()

display_historical_review(
    frames,
    analyses,
)

# ---------------------------------------------------------
# VIRTUAL ACCOUNT
# ---------------------------------------------------------

st.divider()
st.header("Virtual Practice Account")

try:
    account = load_account(
        supabase,
        user_id,
    )

    balance = float(
        account.get("balance", 0)
    )

    transactions = account.get(
        "transactions",
        [],
    )

    account_col, transaction_col = st.columns(2)

    with account_col:
        st.metric(
            "Virtual Balance",
            f"{balance:,.2f}",
        )

    with transaction_col:
        st.metric(
            "Recorded Virtual Transactions",
            len(transactions),
        )

    if transactions:
        st.subheader("Virtual Transaction History")

        transaction_frame = pd.DataFrame(
            transactions
        )

        st.dataframe(
            transaction_frame,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No virtual transactions have been recorded yet."
        )

except Exception as exc:
    st.warning(
        f"Virtual account information unavailable: {exc}"
    )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "DESTINY AI | Historical analysis and virtual review"
)

st.caption(
    "Historical data may contain delays, gaps, or "
    "provider-specific limitations. Virtual results "
    "are simulations and are not actual financial results."
        )
