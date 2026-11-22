from mtf_context import build_mtf_context
from strategy_engine import analyze_market
from sr_engine import analyze_sr
from auth_guard import require_login

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import urllib.request


# =========================================================
# DESTINY AI — HISTORICAL ANALYSIS & VIRTUAL TESTING
# No broker connection or live order execution.
# =========================================================

st.set_page_config(
    page_title="Destiny AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)
require_login()

TIMEFRAMES = ["1D", "4H", "1H", "15M", "5M", "1M"]
SR_TIMEFRAMES = ["4H", "1H", "15M"]

SR_CLASSIFICATION = {
    "4H": "MAJOR",
    "1H": "MAJOR",
    "15M": "MINOR"
}

MARKETS = {
    "Crypto": {
        "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD",
        "Solana": "SOL-USD", "XRP": "XRP-USD",
        "BNB": "BNB-USD", "Litecoin": "LTC-USD",
        "Dogecoin": "DOGE-USD"
    },
    "Forex": {
        "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X",
        "USD/JPY": "USDJPY=X", "GBP/JPY": "GBPJPY=X",
        "EUR/JPY": "EURJPY=X", "AUD/USD": "AUDUSD=X",
        "USD/CAD": "USDCAD=X", "USD/CHF": "USDCHF=X",
        "NZD/USD": "NZDUSD=X"
    },
    "Commodities": {
        "Gold": "GC=F", "Silver": "SI=F",
        "Crude Oil": "CL=F", "Brent Oil": "BZ=F",
        "Natural Gas": "NG=F", "Platinum": "PL=F",
        "Copper": "HG=F"
    }
}


# =========================================================
# OPTIONAL ANDROID NOTIFICATION
# =========================================================

def send_app_notification(title, message):
    try:
        topic = st.secrets.get("NTFY_TOPIC", "").strip()
        if not topic:
            return False

        request = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": title},
            method="POST"
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status in (200, 201)
    except Exception:
        return False


if st.sidebar.button("Test Android Notification"):
    if send_app_notification(
        "Destiny AI",
        "Your Destiny AI notification connection is working."
    ):
        st.sidebar.success("Test notification sent.")
    else:
        st.sidebar.warning(
            "Notification unavailable. Check NTFY_TOPIC in Secrets."
        )


# =========================================================
# HISTORICAL MARKET DATA
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_data(symbol, interval, period):
    try:
        df = yf.Ticker(symbol).history(
            period=period,
            interval=interval,
            auto_adjust=False,
            actions=False,
            prepost=False
        )
        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close"]
        if not all(c in df.columns for c in required):
            return pd.DataFrame()

        if "Volume" not in df.columns:
            df["Volume"] = 0

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

        for column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df = df.dropna(subset=required).sort_index()
        return df[~df.index.duplicated(keep="last")]

    except Exception:
        return pd.DataFrame()


def remove_latest_candle(df):
    if df is None or df.empty or len(df) < 2:
        return pd.DataFrame()

    return df.iloc[:-1].copy()


def build_four_hour_data(hourly):
    if hourly is None or hourly.empty:
        return pd.DataFrame()

    try:
        result = hourly.resample(
            "4h",
            label="right",
            closed="right"
        ).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        })

        result = result.dropna(
            subset=["Open", "High", "Low", "Close"]
        )

        return result.iloc[:-1].copy() if len(result) > 1 else result

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_all_timeframes(symbol):
    daily = fetch_market_data(symbol, "1d", "2y")

    if not daily.empty:
        today_utc = pd.Timestamp.now(tz="UTC").date()
        daily_last = pd.Timestamp(daily.index[-1])

        if daily_last.date() >= today_utc:
            daily = daily.iloc[:-1].copy()

    hourly = remove_latest_candle(
        fetch_market_data(symbol, "1h", "730d")
    )

    fifteen = remove_latest_candle(
        fetch_market_data(symbol, "15m", "60d")
    )

    five = remove_latest_candle(
        fetch_market_data(symbol, "5m", "60d")
    )

    one = remove_latest_candle(
        fetch_market_data(symbol, "1m", "7d")
    )

    return {
        "1D": daily,
        "4H": build_four_hour_data(hourly),
        "1H": hourly,
        "15M": fifteen,
        "5M": five,
        "1M": one
    }


# =========================================================
# ANALYSIS ENGINES
# =========================================================

@st.cache_data(ttl=1800, show_spinner=False)
def run_all_analyses(frames):
    results = {}

    for timeframe in TIMEFRAMES:
        data = frames.get(timeframe, pd.DataFrame())

        if data is None or data.empty:
            results[timeframe] = {
                "status": "UNAVAILABLE",
                "error": "Market data is unavailable."
            }
            continue

        try:
            output = analyze_market(data.tail(1200).copy())

            if not isinstance(output, dict):
                raise ValueError(
                    "Analysis engine returned an invalid result."
                )

            results[timeframe] = {
                "status": "ANALYSED",
                "data": output,
                "error": None
            }

        except Exception as error:
            results[timeframe] = {
                "status": "ERROR",
                "error": str(error)
            }

    return results


@st.cache_data(ttl=1800, show_spinner=False)
def run_sr_analysis(data, source_timeframe):
    """
    Analyse each S/R source timeframe independently.

    Uses completed historical candles, close-based invalidation,
    and the full available history supplied by the data loader.
    """

    if data is None or data.empty:
        return pd.DataFrame()

    clean = data.copy()
    clean = clean.sort_index()
    clean = clean[~clean.index.duplicated(keep="last")]

    required = ["High", "Low", "Close"]

    if not all(column in clean.columns for column in required):
        raise ValueError(
            f"{source_timeframe}: Missing required OHLC columns."
        )

    clean = clean.dropna(subset=required)

    if len(clean) < 30:
        return pd.DataFrame()

    zones = analyze_sr(
        data=clean,
        pivot_length=8,
        zone_atr=1.0,
        atr_length=14,
        strength=3,
        invalid_mode="Close",
        max_zones=40
    )

    if not isinstance(zones, pd.DataFrame):
        raise TypeError(
            f"{source_timeframe}: S/R engine returned an invalid result."
        )

    if zones.empty:
        return pd.DataFrame()

    zones = zones.copy()

    if "Confirmed Time" not in zones.columns:
        raise ValueError(
            f"{source_timeframe}: S/R output is missing Confirmed Time."
        )

    zones["Source Timeframe"] = source_timeframe
    zones["Classification"] = SR_CLASSIFICATION.get(
        source_timeframe,
        "UNCLASSIFIED"
    )

    return zones.sort_values(
        "Confirmed Time",
        ascending=False
    ).head(8).reset_index(drop=True)


def build_combined_sr(frames):
    """
    Always combines 4H major, 1H major and 15M minor S/R.
    This is independent of the main chart timeframe.
    """

    layer_frames = []

    for timeframe in SR_TIMEFRAMES:
        source_data = frames.get(timeframe, pd.DataFrame())
        layer = run_sr_analysis(source_data, timeframe)

        if isinstance(layer, pd.DataFrame) and not layer.empty:
            layer_frames.append(layer)

    if not layer_frames:
        return pd.DataFrame()

    return pd.concat(layer_frames, ignore_index=True)


# =========================================================
# TIME ALIGNMENT
# =========================================================

def align_timestamp(value, reference):
    timestamp = pd.Timestamp(value)

    if reference.tzinfo is None and timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)

    elif reference.tzinfo is not None and timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(reference.tzinfo)

    return timestamp


# =========================================================
# SUPPORT / RESISTANCE DRAWING
# =========================================================

def draw_sr_zones(fig, zones, data, start_time, end_time):
    required = ["Side", "Zone Low", "Zone High"]

    if not isinstance(zones, pd.DataFrame) or zones.empty:
        return

    if not all(column in zones.columns for column in required):
        return

    latest_close = float(data["Close"].iloc[-1])
    valid_zones = []

    for _, zone in zones.iterrows():
        try:
            side = str(zone["Side"]).upper()
            low = float(zone["Zone Low"])
            high = float(zone["Zone High"])

            source_tf = str(
                zone.get("Source Timeframe", "S/R")
            )

            classification = str(
                zone.get("Classification", "UNCLASSIFIED")
            )

            if (
                side not in {"SUPPORT", "RESISTANCE"}
                or pd.isna(low)
                or pd.isna(high)
                or low >= high
            ):
                continue

            valid_zones.append((
                abs(((low + high) / 2) - latest_close),
                zone,
                side,
                low,
                high,
                source_tf,
                classification
            ))

        except Exception:
            continue

    # Keep every valid layer visible; label only the three nearest zones.
    nearest = sorted(
        valid_zones,
        key=lambda item: item[0]
    )[:3]

    label_keys = {
        (
            item[5],
            item[2],
            item[3],
            item[4],
            str(item[1].get("Confirmed Time", ""))
        )
        for item in nearest
    }

    label_index = 0

    for _, zone, side, low, high, source_tf, classification in valid_zones:
        try:
            zone_time = zone.get(
                "Confirmed Time",
                zone.get("Pivot Time", data.index[0])
            )

            if pd.isna(zone_time):
                zone_time = data.index[0]

            zone_time = align_timestamp(zone_time, start_time)
            zone_start = max(zone_time, start_time)
            zone_end = end_time

            if zone_start > zone_end:
                continue

            if side == "SUPPORT":
                color = "#00C087"
            else:
                color = "#F04452"

            # 4H = solid, 1H = dashed, 15M = dotted.
            if source_tf == "4H":
                dash_style = "solid"
                fill_alpha = "0.105"
            elif source_tf == "1H":
                dash_style = "dash"
                fill_alpha = "0.075"
            else:
                dash_style = "dot"
                fill_alpha = "0.045"

            if side == "SUPPORT":
                fill = f"rgba(0,192,135,{fill_alpha})"
            else:
                fill = f"rgba(240,68,82,{fill_alpha})"

            midpoint = (low + high) / 2

            fig.add_shape(
                type="rect",
                x0=zone_start,
                x1=zone_end,
                y0=low,
                y1=high,
                fillcolor=fill,
                line=dict(
                    color=color,
                    width=1.2 if source_tf == "4H" else 0.9,
                    dash=dash_style
                ),
                layer="below"
            )

            fig.add_shape(
                type="line",
                x0=zone_start,
                x1=zone_end,
                y0=midpoint,
                y1=midpoint,
                line=dict(
                    color=color,
                    width=0.8,
                    dash=dash_style
                ),
                layer="below"
            )

            zone_key = (
                source_tf,
                side,
                low,
                high,
                str(zone.get("Confirmed Time", ""))
            )

            if zone_key in label_keys:
                label = f"{source_tf} {classification} · {side}"

                fig.add_annotation(
                    x=end_time,
                    y=midpoint,
                    text=label,
                    showarrow=False,
                    xanchor="right",
                    yanchor="middle",
                    yshift=(label_index - 1) * 12,
                    font=dict(size=8, color=color),
                    bgcolor="rgba(11,14,17,0.82)",
                    borderpad=2
                )

                label_index += 1

        except Exception:
            continue


# =========================================================
# QUALIFIED OB + FVG SHARED AREA
# =========================================================

def draw_ob_fvg_overlaps(fig, setups, data, start_time, end_time):
    required = [
        "ob_time", "ob_low", "ob_high",
        "fvg_time", "fvg_low", "fvg_high", "direction"
    ]

    if not isinstance(setups, pd.DataFrame) or setups.empty:
        return 0

    if not all(column in setups.columns for column in required):
        return 0

    drawn = 0
    active_labels = 0

    for _, setup in setups.tail(100).iterrows():
        if drawn >= 5:
            break

        try:
            ob_time = setup["ob_time"]
            fvg_time = setup["fvg_time"]

            if pd.isna(ob_time) or pd.isna(fvg_time):
                continue

            ob_time = align_timestamp(ob_time, start_time)
            fvg_time = align_timestamp(fvg_time, start_time)
            overlap_time = max(ob_time, fvg_time)

            if overlap_time > end_time:
                continue

            ob_low = float(setup["ob_low"])
            ob_high = float(setup["ob_high"])
            fvg_low = float(setup["fvg_low"])
            fvg_high = float(setup["fvg_high"])

            overlap_low = max(ob_low, fvg_low)
            overlap_high = min(ob_high, fvg_high)

            if overlap_low >= overlap_high:
                continue

            direction = str(setup["direction"]).upper()

            if direction == "BULLISH":
                color = "#00C087"
                invalid = (
                    (data.index >= overlap_time)
                    & (data["Close"] < max(ob_low, fvg_low))
                )

            elif direction == "BEARISH":
                color = "#F04452"
                invalid = (
                    (data.index >= overlap_time)
                    & (data["Close"] > min(ob_high, fvg_high))
                )

            else:
                continue

            invalid_rows = data.loc[invalid]
            setup_end = end_time
            status = str(setup.get("status", "ACTIVE")).upper()

            if not invalid_rows.empty:
                invalid_time = align_timestamp(
                    invalid_rows.index[0],
                    start_time
                )

                if invalid_time < start_time:
                    continue

                setup_end = min(invalid_time, end_time)
                status = "INVALIDATED"

            elif status == "INVALIDATED":
                continue

            visible_start = max(overlap_time, start_time)

            if visible_start > setup_end:
                continue

            invalidated = status == "INVALIDATED"

            if invalidated:
                if direction == "BEARISH":
                    fill = "rgba(240,68,82,0.035)"
                else:
                    fill = "rgba(0,192,135,0.035)"
            else:
                if direction == "BEARISH":
                    fill = "rgba(240,68,82,0.12)"
                else:
                    fill = "rgba(0,192,135,0.12)"

            fig.add_shape(
                type="rect",
                x0=visible_start,
                x1=setup_end,
                y0=overlap_low,
                y1=overlap_high,
                fillcolor=fill,
                line=dict(
                    color=color,
                    width=0.7 if invalidated else 1.0,
                    dash="dot" if invalidated else "solid"
                ),
                layer="below"
            )

            if not invalidated and active_labels < 2:
                label = (
                    "BULL OB/FVG"
                    if direction == "BULLISH"
                    else "BEAR OB/FVG"
                )

                fig.add_annotation(
                    x=end_time,
                    y=(overlap_low + overlap_high) / 2,
                    text=label,
                    showarrow=False,
                    xanchor="right",
                    yanchor="middle",
                    yshift=active_labels * 12,
                    font=dict(size=8, color=color),
                    bgcolor="rgba(11,14,17,0.82)",
                    borderpad=2
                )

                active_labels += 1

            drawn += 1

        except Exception:
            continue

    return drawn


# =========================================================
# MOBILE-FRIENDLY PRICE CHART
# =========================================================

def display_chart(data, result, sr_zones, symbol, timeframe):
    chart_data = data.tail(1200).copy()

    if chart_data.empty:
        st.warning("No candles are available.")
        return

    x_control, y_control = st.columns(2)

    with x_control:
        visible_count = st.slider(
            "Candles on screen",
            min_value=30,
            max_value=180,
            value=60,
            step=10,
            key=f"destiny_visible_candles_{symbol}_{timeframe}"
        )

    with y_control:
        vertical_scale = st.slider(
            "Vertical zoom",
            min_value=0.6,
            max_value=2.0,
            value=1.0,
            step=0.1,
            key=f"destiny_vertical_zoom_{symbol}_{timeframe}",
            help=(
                "Lower values zoom closer vertically; higher values "
                "show more price range."
            )
        )

    start_time = chart_data.index[0]
    end_time = chart_data.index[-1]
    latest_close = float(chart_data["Close"].iloc[-1])

    visible_data = chart_data.tail(visible_count)
    first_visible = visible_data.index[0]

    visible_low = float(visible_data["Low"].min())
    visible_high = float(visible_data["High"].max())

    visible_span = max(
        visible_high - visible_low,
        abs(latest_close) * 0.002
    )

    center_price = (visible_high + visible_low) / 2

    half_range = max(
        visible_span * 0.55 * vertical_scale,
        abs(latest_close) * 0.0005
    )

    initial_y_range = [
        center_price - half_range,
        center_price + half_range
    ]

    volume_data = (
        chart_data["Volume"]
        .fillna(0)
        .to_numpy()
        .reshape(-1, 1)
    )

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=chart_data.index,
        open=chart_data["Open"],
        high=chart_data["High"],
        low=chart_data["Low"],
        close=chart_data["Close"],
        customdata=volume_data,
        name="Price",
        increasing_line_color="#20C997",
        decreasing_line_color="#F05B65",
        increasing_fillcolor="#20C997",
        decreasing_fillcolor="#F05B65",
        whiskerwidth=1.0,
        hovertemplate=(
            "<b>%{x|%d %b %Y %H:%M}</b><br>"
            "Open: %{open:,.5f}<br>"
            "High: %{high:,.5f}<br>"
            "Low: %{low:,.5f}<br>"
            "Close: %{close:,.5f}<br>"
            "Volume: %{customdata[0]:,.0f}"
            "<extra></extra>"
        )
    ))

    draw_sr_zones(
        fig,
        sr_zones,
        chart_data,
        start_time,
        end_time
    )

    overlap_count = draw_ob_fvg_overlaps(
        fig,
        result.get("setups", pd.DataFrame()),
        chart_data,
        start_time,
        end_time
    )

    liquidity = result.get("liquidity", pd.DataFrame())

    if (
        isinstance(liquidity, pd.DataFrame)
        and not liquidity.empty
        and {"swept", "price", "side"}.issubset(liquidity.columns)
    ):
        active = liquidity[liquidity["swept"] == False].tail(6)

        for _, level in active.iterrows():
            try:
                price = float(level["price"])

                color = (
                    "#F05B65"
                    if str(level["side"]).upper() == "BSL"
                    else "#20C997"
                )

                fig.add_hline(
                    y=price,
                    line_dash="dot",
                    line_color=color,
                    line_width=1.2
                )

            except Exception:
                continue

    fig.add_hline(
        y=latest_close,
        line_color="#FFD54F",
        line_width=1.7,
        annotation_text=f"Latest close {latest_close:,.5f}",
        annotation_position="top right",
        annotation_font=dict(color="#FFD54F", size=10),
        annotation_bgcolor="rgba(20,20,20,0.88)"
    )

    reset_id = st.session_state.get(
        "destiny_chart_reset_counter",
        0
    )

    fig.update_layout(
        title=dict(
            text=f"{symbol} | {timeframe}",
            x=0.02,
            xanchor="left",
            font=dict(size=18, color="#E6EAF0")
        ),
        template="plotly_dark",
        paper_bgcolor="#0B0E11",
        plot_bgcolor="#0B0E11",
        height=640,
        autosize=True,
        margin=dict(l=8, r=82, t=58, b=35),
        font=dict(
            family="Arial, sans-serif",
            size=13,
            color="#D7DCE2"
        ),
        hoverlabel=dict(
            bgcolor="#20262D",
            bordercolor="#59636E",
            font=dict(size=12, color="#FFFFFF")
        ),
        hovermode="closest",
        dragmode="pan",
        showlegend=False,
        uirevision=f"destiny-{symbol}-{timeframe}-{reset_id}",
        xaxis=dict(
            type="date",
            range=[first_visible, end_time],
            rangeslider=dict(visible=False),
            tickformat="%d %b",
            nticks=5,
            showgrid=True,
            gridcolor="rgba(180,190,200,0.10)",
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            automargin=True
        ),
        yaxis=dict(
            side="right",
            range=initial_y_range,
            autorange=False,
            fixedrange=False,
            showgrid=True,
            gridcolor="rgba(180,190,200,0.10)",
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            automargin=True
        )
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
                "resetScale2d"
            ],
            "modeBarButtonsToRemove": [
                "select2d",
                "lasso2d"
            ]
        },
        key=f"destiny-chart-{symbol}-{timeframe}"
    )

    st.caption(
        "4H major S/R = solid outline | 1H major S/R = dashed outline | "
        "15M minor S/R = dotted outline. Green = support / bullish; "
        "red = resistance / bearish. S/R zones remain valid until a "
        "completed candle on their own source timeframe closes beyond "
        "the outer boundary. Yellow = latest completed close."
    )

    if overlap_count == 0:
        st.info(
            "No qualified OB + FVG overlap is visible in this chart window. "
            "Standalone FVGs are hidden."
        )


# =========================================================
# STATISTICS AND SCANNER
# =========================================================

def get_statistics(results):
    rows = []

    for timeframe in TIMEFRAMES:
        item = results.get(timeframe, {})

        if item.get("status") != "ANALYSED":
            rows.append({
                "Timeframe": timeframe,
                "Status": item.get("status", "UNKNOWN"),
                "Candles": 0,
                "Bullish BOS": 0,
                "Bearish BOS": 0,
                "Qualified Setups": 0,
                "Active Setups": 0
            })
            continue

        result = item["data"]
        stats = result.get("statistics", {})

        rows.append({
            "Timeframe": timeframe,
            "Status": "ANALYSED",
            "Candles": len(result.get("candles", [])),
            "Bullish BOS": stats.get("bullish_bos", 0),
            "Bearish BOS": stats.get("bearish_bos", 0),
            "Qualified Setups": stats.get("total_qualified_setups", 0),
            "Active Setups": stats.get("active_setups", 0)
        })

    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner=False)
def scan_market_category(category):
    rows = []

    for instrument, ticker in MARKETS.get(category, {}).items():
        try:
            history = remove_latest_candle(
                fetch_market_data(ticker, "1h", "60d")
            )

            if history.empty or len(history) < 40:
                rows.append({
                    "Instrument": instrument,
                    "Symbol": ticker,
                    "Data status": "Unavailable / insufficient",
                    "Candles analysed": len(history),
                    "Bullish BOS history": "—",
                    "Bearish BOS history": "—",
                    "Historical OB + FVG setups": "—",
                    "Active historical setups": "—"
                })
                continue

            analysis = analyze_market(history.tail(1200).copy())
            stats = analysis.get("statistics", {})

            rows.append({
                "Instrument": instrument,
                "Symbol": ticker,
                "Data status": "Analysed",
                "Candles analysed": len(analysis.get("candles", [])),
                "Bullish BOS history": int(
                    stats.get("bullish_bos", 0) or 0
                ),
                "Bearish BOS history": int(
                    stats.get("bearish_bos", 0) or 0
                ),
                "Historical OB + FVG setups": int(
                    stats.get("total_qualified_setups", 0) or 0
                ),
                "Active historical setups": int(
                    stats.get("active_setups", 0) or 0
                )
            })

        except Exception:
            rows.append({
                "Instrument": instrument,
                "Symbol": ticker,
                "Data status": "Analysis unavailable",
                "Candles analysed": 0,
                "Bullish BOS history": "—",
                "Bearish BOS history": "—",
                "Historical OB + FVG setups": "—",
                "Active historical setups": "—"
            })

    return pd.DataFrame(rows)


# =========================================================
# APP HEADER AND WORKSPACE
# =========================================================

st.title("DESTINY AI")
st.caption("Multi-timeframe historical market analysis")

st.info(
    "Educational historical analysis and virtual paper-testing only. "
    "No broker connection or transaction execution."
)

workspace = st.radio(
    "Choose workspace",
    ["Main Analysis", "Historical Market Scanner"],
    horizontal=True,
    key="destiny_workspace"
)


# =========================================================
# HISTORICAL SCANNER
# =========================================================

if workspace == "Historical Market Scanner":
    st.divider()
    st.subheader("Historical Market Scanner")

    st.caption(
        "Compares completed 1-hour historical candles. "
        "Results describe past data, not forecasts."
    )

    category = st.selectbox(
        "Market category",
        list(MARKETS.keys()),
        key="scanner_market_category"
    )

    left, right = st.columns(2)

    with left:
        run_scan = st.button(
            "SCAN SELECTED MARKET",
            use_container_width=True,
            type="primary"
        )

    with right:
        refresh_scan = st.button(
            "REFRESH SCANNER DATA",
            use_container_width=True
        )

    if refresh_scan:
        scan_market_category.clear()
        fetch_market_data.clear()
        st.rerun()

    if (
        run_scan
        or "destiny_scanner_results" not in st.session_state
        or st.session_state.get("destiny_scanner_category") != category
    ):
        with st.spinner(f"Reviewing historical {category} data..."):
            st.session_state["destiny_scanner_results"] = (
                scan_market_category(category)
            )
            st.session_state["destiny_scanner_category"] = category

    scan_results = st.session_state.get(
        "destiny_scanner_results",
        pd.DataFrame()
    )

    if (
        isinstance(scan_results, pd.DataFrame)
        and not scan_results.empty
    ):
        count = int(
            (scan_results["Data status"] == "Analysed").sum()
        )

        st.metric(
            "Instruments analysed",
            f"{count} / {len(scan_results)}"
        )

        st.dataframe(
            scan_results,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download scanner results",
            data=scan_results.to_csv(index=False).encode("utf-8"),
            file_name=f"destiny_{category.lower()}_historical_scan.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:
        st.info(
            "Choose a category and tap SCAN SELECTED MARKET."
        )

    st.caption(
        "Coverage: 1-hour candles, up to 60 days where available."
    )
    st.stop()


# =========================================================
# MARKET SELECTION AND DATA LOAD
# =========================================================

col1, col2 = st.columns(2)

with col1:
    market_type = st.selectbox(
        "Select market",
        list(MARKETS.keys())
    )

with col2:
    asset_name = st.selectbox(
        "Select instrument",
        list(MARKETS[market_type].keys())
    )

symbol = MARKETS[market_type][asset_name]

chart_timeframe = st.selectbox(
    "Main chart timeframe",
    TIMEFRAMES,
    index=2
)

if st.button(
    "REFRESH MARKET DATA",
    use_container_width=True
):
    st.cache_data.clear()
    st.rerun()

try:
    with st.spinner("Loading market data..."):
        frames = load_all_timeframes(symbol)

except Exception as error:
    st.error("Market data could not be loaded.")
    st.caption(str(error))
    st.stop()

try:
    with st.spinner("Analysing available timeframes..."):
        analyses = run_all_analyses(frames)

except Exception as error:
    st.error("The analysis process encountered an error.")
    st.caption(str(error))
    st.stop()

selected_data = frames.get(
    chart_timeframe,
    pd.DataFrame()
)

# Calculate all three S/R layers regardless of selected chart timeframe.
with st.spinner(
    "Building 4H major, 1H major and 15M minor S/R..."
):
    try:
        sr_all = build_combined_sr(frames)
        sr_error = None

    except Exception as error:
        sr_all = pd.DataFrame()
        sr_error = str(error)


# =========================================================
# DATA AVAILABILITY
# =========================================================

st.divider()
st.subheader("Market data availability")

status_columns = st.columns(3)

for index, timeframe in enumerate(TIMEFRAMES):
    frame = frames.get(timeframe, pd.DataFrame())

    with status_columns[index % 3]:
        if not frame.empty:
            st.success(
                f"{timeframe}: {len(frame)} candles"
            )
        else:
            st.warning(
                f"{timeframe}: Unavailable"
            )


# =========================================================
# MAIN CHART — NO TIMED REDRAWS WHILE TOUCHING
# =========================================================

st.divider()
st.subheader("Price chart")

st.caption(
    "The chart opens on the latest 60 candles. Pan through the latest "
    "1,200 completed candles. All three S/R layers remain on the chart "
    "when you change the chart timeframe."
)

if st.button(
    "RETURN TO LATEST",
    use_container_width=True
):
    st.session_state["destiny_chart_reset_counter"] = (
        st.session_state.get(
            "destiny_chart_reset_counter",
            0
        ) + 1
    )
    st.rerun()

selected_analysis = analyses.get(
    chart_timeframe,
    {}
)

if (
    not selected_data.empty
    and selected_analysis.get("status") == "ANALYSED"
):
    display_chart(
        selected_data,
        selected_analysis["data"],
        sr_all,
        symbol,
        chart_timeframe
    )

    st.metric(
        "Latest completed candle close",
        f"{float(selected_data['Close'].iloc[-1]):,.5f}"
    )

    st.download_button(
        "Download historical OHLCV data",
        data=selected_data.to_csv().encode("utf-8"),
        file_name=f"{symbol}_{chart_timeframe}.csv",
        mime="text/csv"
    )

elif selected_data.empty:
    st.warning(
        f"No genuine {chart_timeframe} data is currently available."
    )

else:
    st.error(
        selected_analysis.get(
            "error",
            "Analysis could not be completed."
        )
    )


# =========================================================
# MULTI-TIMEFRAME CONTEXT
# =========================================================

st.divider()
st.subheader("Destiny AI — MTF Structure Context")

try:
    mtf_context = build_mtf_context(analyses)

    st.caption(
        mtf_context.get(
            "data_coverage",
            "MTF coverage unavailable"
        )
    )

    st.markdown("### Higher Timeframe Context")

    st.info(
        mtf_context.get(
            "higher_timeframe_summary",
            "Higher timeframe analysis unavailable."
        )
    )

    st.markdown("### Lower Timeframe Context")

    st.info(
        mtf_context.get(
            "lower_timeframe_summary",
            "Lower timeframe analysis unavailable."
        )
    )

    mtf_rows = []

    for timeframe in TIMEFRAMES:
        item = mtf_context.get(
            "timeframes",
            {}
        ).get(timeframe, {})

        mtf_rows.append({
            "Timeframe": timeframe,
            "Status": item.get("status", "Unavailable"),
            "Bullish BOS": item.get("bullish_bos", 0),
            "Bearish BOS": item.get("bearish_bos", 0),
            "Structure": item.get(
                "structure",
                "Insufficient data"
            )
        })

    st.dataframe(
        pd.DataFrame(mtf_rows),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "MTF context summarizes historical BOS counts. "
        "It does not predict price direction."
    )

except Exception as error:
    st.error("MTF context could not be generated.")
    st.caption(str(error))


# =========================================================
# SUPPORT AND RESISTANCE TABLE
# =========================================================

st.divider()
st.subheader("Historical Support & Resistance Zones")

if sr_error:
    st.error(
        "The S/R calculation encountered an error."
    )
    st.code(sr_error)

if isinstance(sr_all, pd.DataFrame) and not sr_all.empty:
    st.caption(
        "Combined S/R: 4H major, 1H major and 15M minor. "
        "Each layer is calculated from its own completed-candle history. "
        "A zone is invalidated only by a completed candle close beyond "
        "its outer boundary on that zone's source timeframe."
    )

    display_columns = [
        column
        for column in [
            "Source Timeframe",
            "Classification",
            "Side",
            "Zone Low",
            "Zone High",
            "Midline",
            "Confirmed Time",
            "Pivot Time",
            "Rejections",
            "Strength",
            "Strong"
        ]
        if column in sr_all.columns
    ]

    st.dataframe(
        sr_all[display_columns],
        use_container_width=True,
        hide_index=True
    )

else:
    st.info(
        "No 4H, 1H or 15M historical S/R zones are currently available. "
        "Check the data availability section and any S/R error shown above."
    )


# =========================================================
# MULTI-TIMEFRAME STATISTICS
# =========================================================

st.subheader("Multi-timeframe statistics")

st.dataframe(
    get_statistics(analyses),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# DETAILED TIMEFRAME RESULTS
# =========================================================

st.subheader("Detailed timeframe results")

for timeframe in TIMEFRAMES:
    item = analyses.get(timeframe, {})

    with st.expander(
        f"{timeframe} analysis",
        expanded=(timeframe == "1H")
    ):
        if item.get("status") == "UNAVAILABLE":
            st.warning(item.get("error"))
            continue

        if item.get("status") == "ERROR":
            st.error(item.get("error"))
            continue

        result = item["data"]
        stats = result.get("statistics", {})

        a, b, c = st.columns(3)

        a.metric(
            "Bullish BOS",
            stats.get("bullish_bos", 0)
        )

        b.metric(
            "Bearish BOS",
            stats.get("bearish_bos", 0)
        )

        c.metric(
            "Qualified OB + FVG",
            stats.get("total_qualified_setups", 0)
        )

        st.write(
            "**Active setups:**",
            stats.get("active_setups", 0)
        )

        st.write(
            "**Invalidated setups:**",
            stats.get("invalidated_setups", 0)
        )

        st.write(
            "**Active BSL levels:**",
            stats.get("active_bsl_levels", 0)
        )

        st.write(
            "**Active SSL levels:**",
            stats.get("active_ssl_levels", 0)
        )

        setups = result.get(
            "setups",
            pd.DataFrame()
        )

        if (
            isinstance(setups, pd.DataFrame)
            and not setups.empty
        ):
            st.write(
                "**Qualified OB + FVG overlap history**"
            )

            st.dataframe(
                setups.tail(30),
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info(
                "No qualified OB + FVG overlaps were detected in this "
                "historical window. Standalone FVGs are excluded."
            )

        liquidity = result.get(
            "liquidity",
            pd.DataFrame()
        )

        if (
            isinstance(liquidity, pd.DataFrame)
            and not liquidity.empty
        ):
            st.write(
                "**BSL / SSL liquidity history**"
            )

            st.dataframe(
                liquidity.tail(30),
                use_container_width=True,
                hide_index=True
            )


# =========================================================
# ENGINE INFORMATION
# =========================================================

st.divider()

with st.expander("Data and engine information"):
    st.write(
        "Market data source: Yahoo Finance through yfinance."
    )

    st.write(
        "The 4H timeframe is constructed from 1H candles."
    )

    st.write(
        "The latest potentially incomplete intraday candle is excluded."
    )

    st.write(
        "Each timeframe is analysed independently."
    )

    st.write(
        "The strategy engine requires a qualified BOS, displacement, "
        "an opposing order-block candle, and an overlapping FVG."
    )

    st.write(
        "FVG volume qualification remains enabled in strategy_engine.py. "
        "Markets with missing or zero volume may produce fewer qualified overlaps."
    )

    st.write(
        "The chart displays only the shared price intersection of qualified "
        "OB and FVG zones. Standalone FVGs are not displayed."
    )

    st.write(
        "S/R uses completed-candle CLOSE invalidation on the zone's own "
        "source timeframe. A wick alone does not invalidate a zone."
    )

    st.write(
        "S/R layers: 4H major, 1H major and 15M minor. Each layer analyses "
        "the available completed historical candles from its own source timeframe."
    )

    st.write(
        "The strategy analysis uses up to 1,200 candles per timeframe."
    )

    st.write(
        "No fabricated candles are created when market data is unavailable."
    )

    st.write(
        "The chart renders at most 1,200 candles for mobile performance. "
        "It initially displays the latest 60 candles. Timed redraws are disabled "
        "so touch gestures are not interrupted."
    )

st.caption(
    "DESTINY AI | Historical analysis and virtual paper-testing"
)
