from mtf_context import build_mtf_context
from strategy_engine import analyze_market

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import urllib.request


# =========================================================
# DESTINY AI - APPLICATION CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Destiny AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

TIMEFRAMES = ["1D", "4H", "1H", "15M", "5M", "1M"]

MARKETS = {
    "Crypto": {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
        "Solana": "SOL-USD",
        "XRP": "XRP-USD",
        "BNB": "BNB-USD",
        "Litecoin": "LTC-USD",
        "Dogecoin": "DOGE-USD"
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
        "NZD/USD": "NZDUSD=X"
    },
    "Commodities": {
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Crude Oil": "CL=F",
        "Brent Oil": "BZ=F",
        "Natural Gas": "NG=F",
        "Platinum": "PL=F",
        "Copper": "HG=F"
    }
}


# =========================================================
# ANDROID APP STATUS NOTIFICATIONS
# =========================================================

def send_app_notification(title, message):
    """Send a non-trading app-status notification."""

    try:
        topic = st.secrets["NTFY_TOPIC"].strip()

        if not topic:
            st.warning("NTFY_TOPIC is empty. Check Streamlit Secrets.")
            return False

        url = f"https://ntfy.sh/{topic}"

        request = urllib.request.Request(
            url,
            data=message.encode("utf-8"),
            headers={"Title": title},
            method="POST"
        )

        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status in (200, 201)

    except Exception as error:
        st.warning(f"Notification could not be sent: {error}")
        return False


if st.sidebar.button("Test Android Notification"):
    if send_app_notification(
        "Destiny AI",
        "Your Android app-status notification connection is working."
    ):
        st.sidebar.success("Test notification sent. Check your ntfy app.")


# =========================================================
# MARKET DATA DOWNLOAD
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

        if not all(column in df.columns for column in required):
            return pd.DataFrame()

        if "Volume" not in df.columns:
            df["Volume"] = 0

        df = df[
            ["Open", "High", "Low", "Close", "Volume"]
        ].copy()

        for column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

        df = df.dropna(
            subset=["Open", "High", "Low", "Close"]
        )

        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        return df

    except Exception:
        return pd.DataFrame()


def remove_latest_candle(df):

    if df is None or df.empty or len(df) < 2:
        return pd.DataFrame()

    return df.iloc[:-1].copy()


# =========================================================
# FOUR-HOUR DATA CONSTRUCTION
# =========================================================

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

        if len(result) > 1:
            result = result.iloc[:-1]

        return result

    except Exception:
        return pd.DataFrame()


# =========================================================
# LOAD ALL TIMEFRAMES
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def load_all_timeframes(symbol):

    frames = {}

    daily = fetch_market_data(symbol, "1d", "2y")

    if not daily.empty:
        today_utc = pd.Timestamp.now(tz="UTC").date()

        if daily.index[-1].date() >= today_utc:
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

    frames["1D"] = daily
    frames["4H"] = build_four_hour_data(hourly)
    frames["1H"] = hourly
    frames["15M"] = fifteen
    frames["5M"] = five
    frames["1M"] = one

    return frames


# =========================================================
# MULTI-TIMEFRAME STRATEGY ANALYSIS
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
                "error": "Market data is unavailable."
            }

            continue

        try:
            analysis_data = data.tail(1200).copy()

            result = analyze_market(analysis_data)

            if not isinstance(result, dict):
                raise ValueError(
                    "The analysis engine returned an invalid result."
                )

            results[timeframe] = {
                "status": "ANALYSED",
                "data": result,
                "error": None
            }

        except Exception as error:

            results[timeframe] = {
                "status": "ERROR",
                "error": str(error)
            }

    return results


# =========================================================
# SUPPORT AND RESISTANCE ANALYSIS
# =========================================================

@st.cache_data(ttl=1800, show_spinner=False)
def run_sr_analysis(data):

    from sr_engine import analyze_sr

    if data is None or data.empty:
        return pd.DataFrame()

    try:
        sr_input = data.tail(400).copy()

        zones = analyze_sr(sr_input)

        if isinstance(zones, pd.DataFrame):
            return zones.head(8).copy()

        return pd.DataFrame()

    except Exception:
        return pd.DataFrame()


# =========================================================
# TIMESTAMP ALIGNMENT
# =========================================================

def align_timestamp(value, reference):

    timestamp = pd.Timestamp(value)

    if reference.tzinfo is None and timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)

    elif reference.tzinfo is not None and timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(reference.tzinfo)

    return timestamp


# =========================================================
# SUPPORT AND RESISTANCE CHART OVERLAYS
# =========================================================

def draw_sr_zones(fig, zones, start_time, end_time):

    if not isinstance(zones, pd.DataFrame) or zones.empty:
        return

    required_columns = [
        "Side",
        "Zone Low",
        "Zone High"
    ]

    if not all(column in zones.columns for column in required_columns):
        return

    for _, zone in zones.head(8).iterrows():

        try:
            side = str(zone["Side"]).upper()

            zone_low = float(zone["Zone Low"])
            zone_high = float(zone["Zone High"])

            if pd.isna(zone_low) or pd.isna(zone_high):
                continue

            if zone_low >= zone_high:
                continue

            # Use confirmation time when available.
            # This avoids displaying a level before it was confirmed.

            zone_time = zone.get(
                "Confirmed Time",
                zone.get("Pivot Time", start_time)
            )

            if pd.isna(zone_time):
                zone_time = start_time

            zone_time = align_timestamp(
                zone_time,
                start_time
            )

            if zone_time > end_time:
                continue

            zone_start = max(
                zone_time,
                start_time
            )

            if side == "SUPPORT":

                fill = "rgba(0,192,135,0.16)"
                line = "rgba(0,192,135,0.85)"
                label = "SUPPORT"

            elif side == "RESISTANCE":

                fill = "rgba(240,68,82,0.16)"
                line = "rgba(240,68,82,0.85)"
                label = "RESISTANCE"

            else:
                continue

            midpoint = (zone_low + zone_high) / 2

            # Shaded S/R zone.

            fig.add_shape(
                type="rect",
                x0=zone_start,
                x1=end_time,
                y0=zone_low,
                y1=zone_high,
                fillcolor=fill,
                line=dict(
                    color=line,
                    width=1
                ),
                layer="below"
            )

            # 50% zone midpoint.

            fig.add_shape(
                type="line",
                x0=zone_start,
                x1=end_time,
                y0=midpoint,
                y1=midpoint,
                line=dict(
                    color=line,
                    width=1,
                    dash="dash"
                ),
                layer="below"
            )

            # Small label at the right side.

            fig.add_annotation(
                x=end_time,
                y=midpoint,
                text=label,
                showarrow=False,
                xanchor="right",
                yanchor="bottom",
                font=dict(
                    size=9,
                    color=line
                ),
                bgcolor="rgba(20,20,20,0.75)"
            )

        except Exception:
            continue


# =========================================================
# PRICE CHART
# =========================================================

def display_chart(
    data,
    result,
    sr_zones,
    symbol,
    timeframe
):

    chart_data = data.tail(120).copy()

    if chart_data.empty:
        st.warning("No candles are available.")
        return

    start_time = chart_data.index[0]
    end_time = chart_data.index[-1]

    fig = go.Figure()

    # -----------------------------------------------------
    # CANDLESTICKS
    # -----------------------------------------------------

    fig.add_trace(
        go.Candlestick(
            x=chart_data.index,
            open=chart_data["Open"],
            high=chart_data["High"],
            low=chart_data["Low"],
            close=chart_data["Close"],
            name="Candles",
            increasing_line_color="#00C087",
            decreasing_line_color="#F04452",
            increasing_fillcolor="#00C087",
            decreasing_fillcolor="#F04452",
            whiskerwidth=0.8
        )
    )

    # -----------------------------------------------------
    # SUPPORT AND RESISTANCE ZONES
    # -----------------------------------------------------

    draw_sr_zones(
        fig,
        sr_zones,
        start_time,
        end_time
    )

    # -----------------------------------------------------
    # ORDER BLOCK AND FVG OVERLAYS
    # -----------------------------------------------------

    setups = result.get(
        "setups",
        pd.DataFrame()
    )

    required_setup_columns = [
        "ob_time",
        "ob_low",
        "ob_high",
        "direction"
    ]

    if (
        isinstance(setups, pd.DataFrame)
        and not setups.empty
        and all(
            column in setups.columns
            for column in required_setup_columns
        )
    ):

        for _, setup in setups.tail(30).iterrows():

            try:
                ob_time = setup.get("ob_time")

                if pd.isna(ob_time):
                    continue

                ob_time = align_timestamp(
                    ob_time,
                    start_time
                )

                if not start_time <= ob_time <= end_time:
                    continue

                ob_low = float(setup["ob_low"])
                ob_high = float(setup["ob_high"])

                if ob_low >= ob_high:
                    continue

                bullish = (
                    str(setup.get("direction", "")).upper()
                    == "BULLISH"
                )

                zone_color = (
                    "rgba(0,192,135,0.13)"
                    if bullish
                    else "rgba(240,68,82,0.13)"
                )

                fig.add_shape(
                    type="rect",
                    x0=ob_time,
                    x1=end_time,
                    y0=ob_low,
                    y1=ob_high,
                    fillcolor=zone_color,
                    line=dict(width=1),
                    layer="below"
                )

                # FVG OVERLAY

                if all(
                    column in setups.columns
                    for column in [
                        "fvg_time",
                        "fvg_low",
                        "fvg_high"
                    ]
                ):

                    fvg_time = setup.get("fvg_time")

                    if pd.notna(fvg_time):

                        fvg_time = align_timestamp(
                            fvg_time,
                            start_time
                        )

                        fvg_low = float(setup["fvg_low"])
                        fvg_high = float(setup["fvg_high"])

                        if (
                            start_time <= fvg_time <= end_time
                            and fvg_low < fvg_high
                        ):

                            fig.add_shape(
                                type="rect",
                                x0=fvg_time,
                                x1=end_time,
                                y0=fvg_low,
                                y1=fvg_high,
                                fillcolor="rgba(245,190,55,0.13)",
                                line=dict(
                                    color="rgba(245,190,55,0.8)",
                                    width=1
                                ),
                                layer="below"
                            )

            except Exception:
                continue

    # -----------------------------------------------------
    # BSL AND SSL LIQUIDITY LEVELS
    # -----------------------------------------------------

    liquidity = result.get(
        "liquidity",
        pd.DataFrame()
    )

    if (
        isinstance(liquidity, pd.DataFrame)
        and not liquidity.empty
        and "swept" in liquidity.columns
        and "price" in liquidity.columns
    ):

        active = liquidity[
            liquidity["swept"] == False
        ].tail(6)

        for _, level in active.iterrows():

            try:
                price = float(level["price"])

                if pd.isna(price):
                    continue

                side = level.get("side")

                color = (
                    "#F04452"
                    if side == "BSL"
                    else "#00C087"
                )

                fig.add_hline(
                    y=price,
                    line_dash="dot",
                    line_color=color,
                    line_width=1
                )

            except Exception:
                continue

    # -----------------------------------------------------
    # CHART APPEARANCE
    # -----------------------------------------------------

    fig.update_layout(
        title=dict(
            text=f"{symbol} | {timeframe}",
            font=dict(size=17)
        ),
        template="plotly_dark",
        height=460,
        autosize=True,
        margin=dict(
            l=10,
            r=15,
            t=50,
            b=35
        ),
        font=dict(size=13),
        xaxis=dict(
            title="Time",
            type="date",
            range=[start_time, end_time],
            rangeslider=dict(visible=False),
            tickformat="%d %b",
            nticks=6,
            tickfont=dict(size=11),
            showgrid=True,
            gridcolor="rgba(130,130,130,0.15)",
            automargin=True
        ),
        yaxis=dict(
            title="Price",
            side="right",
            autorange=True,
            tickfont=dict(size=12),
            showgrid=True,
            gridcolor="rgba(130,130,130,0.15)",
            automargin=True
        ),
        showlegend=False,
        dragmode="pan"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "responsive": True
        }
    )

    st.caption(
        "Green shaded zones: Support | "
        "Red shaded zones: Resistance | "
        "Dashed centre lines: 50% zone midpoint | "
        "Green OB: Bullish | Red OB: Bearish | "
        "Yellow: FVG | Red dotted: BSL | Green dotted: SSL"
    )


# =========================================================
# MULTI-TIMEFRAME STATISTICS
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


# =========================================================
# HISTORICAL MARKET SCANNER
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def scan_market_category(category):

    rows = []

    for instrument, ticker in MARKETS.get(category, {}).items():

        try:
            history = remove_latest_candle(
                fetch_market_data(
                    ticker,
                    "1h",
                    "60d"
                )
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

            analysis = analyze_market(
                history.tail(1200).copy()
            )

            stats = analysis.get(
                "statistics",
                {}
            )

            rows.append({
                "Instrument": instrument,
                "Symbol": ticker,
                "Data status": "Analysed",
                "Candles analysed": len(
                    analysis.get("candles", [])
                ),
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
# APPLICATION HEADER
# =========================================================

st.title("DESTINY AI")

st.caption(
    "Multi-timeframe historical market analysis"
)

st.info(
    "Educational analysis and paper-testing only. "
    "This application does not execute transactions."
)

workspace = st.radio(
    "Choose workspace",
    [
        "Main Analysis",
        "Historical Market Scanner"
    ],
    horizontal=True,
    key="destiny_workspace"
)


# =========================================================
# HISTORICAL MARKET SCANNER WORKSPACE
# =========================================================

if workspace == "Historical Market Scanner":

    st.divider()
    st.subheader("Historical Market Scanner")

    st.caption(
        "Compares completed 1-hour historical data for instruments "
        "in the selected category. Results are descriptive statistics, "
        "not forecasts."
    )

    scanner_category = st.selectbox(
        "Market category",
        list(MARKETS.keys()),
        key="scanner_market_category"
    )

    scan_col, refresh_col = st.columns(2)

    with scan_col:
        run_scan = st.button(
            "SCAN SELECTED MARKET",
            use_container_width=True,
            type="primary"
        )

    with refresh_col:
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
        or st.session_state.get("destiny_scanner_category")
        != scanner_category
    ):

        with st.spinner(
            f"Reviewing historical {scanner_category} data..."
        ):

            st.session_state["destiny_scanner_results"] = (
                scan_market_category(scanner_category)
            )

            st.session_state["destiny_scanner_category"] = (
                scanner_category
            )

    scanner_results = st.session_state.get(
        "destiny_scanner_results",
        pd.DataFrame()
    )

    if (
        isinstance(scanner_results, pd.DataFrame)
        and not scanner_results.empty
    ):

        analysed_count = int(
            (
                scanner_results["Data status"] == "Analysed"
            ).sum()
        )

        st.metric(
            "Instruments analysed",
            f"{analysed_count} / {len(scanner_results)}"
        )

        st.dataframe(
            scanner_results,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download scanner results",
            data=scanner_results.to_csv(
                index=False
            ).encode("utf-8"),
            file_name=(
                f"destiny_{scanner_category.lower()}_historical_scan.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )

    else:

        st.info(
            "Choose a market category and tap SCAN SELECTED MARKET."
        )

    st.caption(
        "Scanner coverage: 1-hour candles, up to 60 days where "
        "the data provider makes them available. Data is cached "
        "for 5 minutes."
    )

    st.warning(
        "BOS and OB + FVG counts describe past candles only. "
        "They are not predictions or trading instructions."
    )

    st.stop()


# =========================================================
# MARKET SELECTION
# =========================================================

left, right = st.columns(2)

with left:

    market_type = st.selectbox(
        "Select market",
        list(MARKETS.keys())
    )

with right:

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

refresh = st.button(
    "REFRESH MARKET DATA",
    use_container_width=True
)

if refresh:

    st.cache_data.clear()
    st.rerun()


# =========================================================
# LOAD MARKET DATA
# =========================================================

try:

    with st.spinner("Loading market data..."):

        frames = load_all_timeframes(symbol)

except Exception as error:

    st.error("The market data could not be loaded.")
    st.code(str(error))
    st.stop()


# =========================================================
# RUN MULTI-TIMEFRAME ANALYSIS
# =========================================================

try:

    with st.spinner("Analysing available timeframes..."):

        analyses = run_all_analyses(frames)

except Exception as error:

    st.error("The analysis process encountered an error.")
    st.code(str(error))
    st.stop()


# =========================================================
# SELECTED-TIMEFRAME SUPPORT AND RESISTANCE
# =========================================================

selected_data = frames.get(
    chart_timeframe,
    pd.DataFrame()
)

try:

    with st.spinner(
        f"Analysing historical S/R on {chart_timeframe}..."
    ):

        sr_selected = run_sr_analysis(selected_data)

except Exception as error:

    st.error("Historical S/R analysis encountered an error.")
    st.caption(str(error))

    sr_selected = pd.DataFrame()


# =========================================================
# MARKET DATA AVAILABILITY
# =========================================================

st.divider()
st.subheader("Market data availability")

status_columns = st.columns(3)

for index, timeframe in enumerate(TIMEFRAMES):

    data = frames.get(
        timeframe,
        pd.DataFrame()
    )

    with status_columns[index % 3]:

        if not data.empty:

            st.success(
                f"{timeframe}: {len(data)} candles"
            )

        else:

            st.warning(
                f"{timeframe}: Unavailable"
            )


# =========================================================
# MAIN PRICE CHART
# =========================================================

st.divider()
st.subheader("Price chart")

selected_analysis = analyses.get(
    chart_timeframe,
    {}
)

if (
    not selected_data.empty
    and selected_analysis.get("status") == "ANALYSED"
):

    result = selected_analysis["data"]

    display_chart(
        selected_data,
        result,
        sr_selected,
        symbol,
        chart_timeframe
    )

    latest_close = selected_data["Close"].iloc[-1]

    st.metric(
        "Latest completed candle close",
        f"{latest_close:,.5f}"
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
# MULTI-TIMEFRAME CONTEXT ENGINE
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

    mtf_table = pd.DataFrame(mtf_rows)

    st.dataframe(
        mtf_table,
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
# HISTORICAL SUPPORT AND RESISTANCE TABLE
# =========================================================

st.divider()
st.subheader("Historical Support & Resistance Zones")

if (
    isinstance(sr_selected, pd.DataFrame)
    and not sr_selected.empty
):

    st.caption(
        f"Up to 8 historical S/R zones detected on "
        f"{chart_timeframe}. The calculation uses the latest "
        "400 candles."
    )

    st.dataframe(
        sr_selected.head(8),
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No historical S/R zones are available "
        "for this timeframe."
    )


# =========================================================
# MULTI-TIMEFRAME STATISTICS
# =========================================================

summary = get_statistics(analyses)

st.subheader("Multi-timeframe statistics")

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# INDIVIDUAL TIMEFRAME DETAILS
# =========================================================

st.subheader("Detailed timeframe results")

for timeframe in TIMEFRAMES:

    item = analyses.get(
        timeframe,
        {}
    )

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

        stats = result.get(
            "statistics",
            {}
        )

        a, b, c = st.columns(3)

        with a:

            st.metric(
                "Bullish BOS",
                stats.get("bullish_bos", 0)
            )

        with b:

            st.metric(
                "Bearish BOS",
                stats.get("bearish_bos", 0)
            )

        with c:

            st.metric(
                "Qualified setups",
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

            st.write("**Qualified OB + FVG setups**")

            st.dataframe(
                setups.tail(30),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No qualified OB + FVG setups were detected "
                "in the available analysis window."
            )

        liquidity = result.get(
            "liquidity",
            pd.DataFrame()
        )

        if (
            isinstance(liquidity, pd.DataFrame)
            and not liquidity.empty
        ):

            st.write("**BSL / SSL liquidity history**")

            st.dataframe(
                liquidity.tail(30),
                use_container_width=True,
                hide_index=True
            )


# =========================================================
# ENGINE AND DATA INFORMATION
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
        "The latest potentially incomplete intraday candle "
        "is excluded."
    )

    st.write(
        "Each timeframe is analysed independently using "
        "analyze_market()."
    )

    st.write(
        "The strategy engine detects BOS, qualified OB/FVG "
        "overlap, BSL/SSL liquidity and setup invalidation."
    )

    st.write(
        "The detailed strategy analysis uses the latest "
        "1,200 candles per timeframe."
    )

    st.write(
        "The S/R engine calculates zones using the latest "
        "400 candles on the selected chart timeframe."
    )

    st.write(
        "A maximum of 8 S/R zones is displayed."
    )

    st.write(
        "Support and resistance zones are drawn directly "
        "on the candlestick chart."
    )

    st.write(
        "Yahoo Finance may not provide every timeframe "
        "for every instrument."
    )

    st.write(
        "Unavailable data is never replaced with fabricated candles."
    )

st.caption(
    "DESTINY AI | Historical market analysis and paper-testing"
        )
