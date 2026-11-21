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
# DESTINY AI — HISTORICAL ANALYSIS APP
# Educational charting and virtual paper-testing only.
# No broker connection or live order execution.
# =========================================================

st.set_page_config(
    require_login()
    page_title="Destiny AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

TIMEFRAMES = ["1D", "4H", "1H", "15M", "5M", "1M"]

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
# OPTIONAL ANDROID APP-STATUS NOTIFICATION
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

    except Exception as error:
        st.error(f"Notification error: {error}")
        return False


if st.sidebar.button("Test Android Notification"):
    if send_app_notification(
        "Destiny AI",
        "Your Destiny AI app-status notification connection is working."
    ):
        st.sidebar.success("Test notification sent.")
    else:
        st.sidebar.warning("Notification unavailable. Check NTFY_TOPIC in Secrets.")


# =========================================================
# MARKET DATA
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
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df
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
            "4h", label="right", closed="right"
        ).agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum"
        })
        result = result.dropna(subset=["Open", "High", "Low", "Close"])
        if len(result) > 1:
            result = result.iloc[:-1]
        return result
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_all_timeframes(symbol):
    daily = fetch_market_data(symbol, "1d", "2y")
    if not daily.empty:
        today_utc = pd.Timestamp.now(tz="UTC").date()
        if daily.index[-1].date() >= today_utc:
            daily = daily.iloc[:-1].copy()

    hourly = remove_latest_candle(fetch_market_data(symbol, "1h", "730d"))
    fifteen = remove_latest_candle(fetch_market_data(symbol, "15m", "60d"))
    five = remove_latest_candle(fetch_market_data(symbol, "5m", "60d"))
    one = remove_latest_candle(fetch_market_data(symbol, "1m", "7d"))

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
            result = analyze_market(data.tail(1200).copy())
            if not isinstance(result, dict):
                raise ValueError("Analysis engine returned an invalid result.")
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


@st.cache_data(ttl=1800, show_spinner=False)
def run_sr_analysis(data):
    if data is None or data.empty:
        return pd.DataFrame()
    try:
        zones = analyze_sr(data.tail(400).copy())
        if isinstance(zones, pd.DataFrame):
            return zones.head(8).copy()
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


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
# SUPPORT / RESISTANCE OVERLAY
# Invalidates only on a completed candle CLOSE beyond
# the outside edge of the zone. A wick is not enough.
# =========================================================

def draw_sr_zones(fig, zones, data, start_time, end_time):
    if not isinstance(zones, pd.DataFrame) or zones.empty:
        return

    required = ["Side", "Zone Low", "Zone High"]
    if not all(c in zones.columns for c in required):
        return

    for _, zone in zones.head(8).iterrows():
        try:
            side = str(zone["Side"]).upper()
            low = float(zone["Zone Low"])
            high = float(zone["Zone High"])
            if pd.isna(low) or pd.isna(high) or low >= high:
                continue

            zone_time = zone.get(
                "Confirmed Time",
                zone.get("Pivot Time", data.index[0])
            )
            if pd.isna(zone_time):
                zone_time = data.index[0]
            zone_time = align_timestamp(zone_time, start_time)

            after_confirmation = data[data.index >= zone_time]

            if side == "SUPPORT":
                broken = after_confirmation[after_confirmation["Close"] < low]
                fill = "rgba(0,192,135,0.15)"
                line = "#00C087"
            elif side == "RESISTANCE":
                broken = after_confirmation[after_confirmation["Close"] > high]
                fill = "rgba(240,68,82,0.15)"
                line = "#F04452"
            else:
                continue

            invalidated = not broken.empty
            if invalidated:
                break_time = align_timestamp(broken.index[0], start_time)
                if break_time < start_time:
                    continue
                zone_end = min(break_time, end_time)
            else:
                zone_end = end_time

            zone_start = max(zone_time, start_time)
            if zone_start > zone_end:
                continue

            midpoint = (low + high) / 2
            fig.add_shape(
                type="rect", x0=zone_start, x1=zone_end,
                y0=low, y1=high, fillcolor=fill,
                line=dict(color=line, width=1), layer="below"
            )
            fig.add_shape(
                type="line", x0=zone_start, x1=zone_end,
                y0=midpoint, y1=midpoint,
                line=dict(color=line, width=1, dash="dash"),
                layer="below"
            )
            label = f"{side} INVALIDATED" if invalidated else side
            fig.add_annotation(
                x=zone_end, y=midpoint, text=label,
                showarrow=False, xanchor="right", yanchor="bottom",
                font=dict(size=9, color=line),
                bgcolor="rgba(20,20,20,0.85)"
            )
        except Exception:
            continue


# =========================================================
# UNIFIED OB + FVG OVERLAP OVERLAY
# Only the actual shared price area is drawn.
# No standalone FVGs and no separate OB boxes.
# =========================================================

def draw_ob_fvg_overlaps(fig, setups, data, start_time, end_time):
    required = [
        "ob_time", "ob_low", "ob_high",
        "fvg_time", "fvg_low", "fvg_high", "direction"
    ]
    if not isinstance(setups, pd.DataFrame) or setups.empty:
        return 0
    if not all(c in setups.columns for c in required):
        return 0

    drawn = 0

    for _, setup in setups.tail(100).iterrows():
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

            # Intersection of the OB and FVG price ranges.
            overlap_low = max(ob_low, fvg_low)
            overlap_high = min(ob_high, fvg_high)
            if not all(pd.notna(v) for v in [
                ob_low, ob_high, fvg_low, fvg_high
            ]):
                continue
            if overlap_low >= overlap_high:
                continue

            direction = str(setup["direction"]).upper()
            if direction == "BULLISH":
                color = "#00C087"
                fill = "rgba(0,192,135,0.23)"
                invalid_condition = (
                    (data.index >= overlap_time)
                    & (data["Close"] < min(ob_low, fvg_low))
                )
            elif direction == "BEARISH":
                color = "#F04452"
                fill = "rgba(240,68,82,0.23)"
                invalid_condition = (
                    (data.index >= overlap_time)
                    & (data["Close"] > max(ob_high, fvg_high))
                )
            else:
                continue

            # The strategy engine's paired setup is invalidated by
            # a completed close through either paired zone boundary.
            invalid_rows = data.loc[invalid_condition]
            setup_end = end_time
            status = str(setup.get("status", "ACTIVE")).upper()

            if not invalid_rows.empty:
                invalid_time = align_timestamp(invalid_rows.index[0], start_time)
                if invalid_time < start_time:
                    continue
                setup_end = min(invalid_time, end_time)
                status = "INVALIDATED"
            elif status == "INVALIDATED":
                # Status can be historical; don't extend an invalidated
                # setup if the precise invalidation candle is outside view.
                continue

            if overlap_time < start_time:
                # Keep only zones that remain relevant in the visible
                # chart window; start them at the left chart boundary.
                visible_start = start_time
            else:
                visible_start = overlap_time

            if visible_start > setup_end:
                continue

            fig.add_shape(
                type="rect",
                x0=visible_start,
                x1=setup_end,
                y0=overlap_low,
                y1=overlap_high,
                fillcolor=fill,
                line=dict(color=color, width=1.5),
                layer="below"
            )
            label = f"{direction} OB + FVG"
            if status == "INVALIDATED":
                label += " (INVALIDATED)"
            fig.add_annotation(
                x=visible_start,
                y=(overlap_low + overlap_high) / 2,
                text=label,
                showarrow=False,
                xanchor="left",
                font=dict(size=9, color=color),
                bgcolor="rgba(20,20,20,0.85)"
            )
            drawn += 1
        except Exception:
            continue

    return drawn


# =========================================================
# PRICE CHART
# =========================================================

def display_chart(data, result, sr_zones, symbol, timeframe):
    chart_data = data.tail(120).copy()
    if chart_data.empty:
        st.warning("No candles are available.")
        return

    start_time = chart_data.index[0]
    end_time = chart_data.index[-1]
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
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
    ))

    draw_sr_zones(fig, sr_zones, data, start_time, end_time)

    setups = result.get("setups", pd.DataFrame())
    overlap_count = draw_ob_fvg_overlaps(
        fig, setups, data, start_time, end_time
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
                color = "#F04452" if level["side"] == "BSL" else "#00C087"
                fig.add_hline(
                    y=price, line_dash="dot",
                    line_color=color, line_width=1
                )
            except Exception:
                continue

    fig.update_layout(
        title=dict(text=f"{symbol} | {timeframe}", font=dict(size=17)),
        template="plotly_dark",
        height=460,
        autosize=True,
        margin=dict(l=10, r=15, t=50, b=35),
        font=dict(size=13),
        xaxis=dict(
            title="Time", type="date",
            range=[start_time, end_time],
            rangeslider=dict(visible=False),
            tickformat="%d %b", nticks=6,
            tickfont=dict(size=11), showgrid=True,
            gridcolor="rgba(130,130,130,0.15)",
            automargin=True
        ),
        yaxis=dict(
            title="Price", side="right", autorange=True,
            tickfont=dict(size=12), showgrid=True,
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
        "Green S/R = support | Red S/R = resistance | "
        "Dashed S/R centre = 50% midpoint | "
        "Green = bullish OB/FVG shared area | "
        "Red = bearish OB/FVG shared area | "
        "Dotted red = BSL | Dotted green = SSL"
    )

    if overlap_count == 0:
        st.info(
            "No qualified OB + FVG overlap is visible in this chart window. "
            "Standalone FVGs are intentionally hidden."
        )


# =========================================================
# TABLES / STATISTICS
# =========================================================

def get_statistics(results):
    rows = []
    for timeframe in TIMEFRAMES:
        item = results.get(timeframe, {})
        if item.get("status") != "ANALYSED":
            rows.append({
                "Timeframe": timeframe,
                "Status": item.get("status", "UNKNOWN"),
                "Candles": 0, "Bullish BOS": 0, "Bearish BOS": 0,
                "Qualified Setups": 0, "Active Setups": 0
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
                    "Instrument": instrument, "Symbol": ticker,
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
                "Bullish BOS history": int(stats.get("bullish_bos", 0) or 0),
                "Bearish BOS history": int(stats.get("bearish_bos", 0) or 0),
                "Historical OB + FVG setups": int(
                    stats.get("total_qualified_setups", 0) or 0
                ),
                "Active historical setups": int(
                    stats.get("active_setups", 0) or 0
                )
            })
        except Exception:
            rows.append({
                "Instrument": instrument, "Symbol": ticker,
                "Data status": "Analysis unavailable",
                "Candles analysed": 0,
                "Bullish BOS history": "—",
                "Bearish BOS history": "—",
                "Historical OB + FVG setups": "—",
                "Active historical setups": "—"
            })
    return pd.DataFrame(rows)


# =========================================================
# APPLICATION HEADER / WORKSPACE
# =========================================================

st.title("DESTINY AI")
st.caption("Multi-timeframe historical market analysis")
st.info(
    "Educational historical analysis and virtual paper-testing only. "
    "This app does not connect to a broker or execute transactions."
)

workspace = st.radio(
    "Choose workspace",
    ["Main Analysis", "Historical Market Scanner"],
    horizontal=True,
    key="destiny_workspace"
)


# =========================================================
# HISTORICAL MARKET SCANNER
# =========================================================

if workspace == "Historical Market Scanner":
    st.divider()
    st.subheader("Historical Market Scanner")
    st.caption(
        "Compares completed 1-hour historical candles. "
        "Results describe past data and are not forecasts."
    )

    scanner_category = st.selectbox(
        "Market category", list(MARKETS.keys()),
        key="scanner_market_category"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        run_scan = st.button(
            "SCAN SELECTED MARKET",
            use_container_width=True,
            type="primary"
        )
    with col_b:
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
        or st.session_state.get("destiny_scanner_category") != scanner_category
    ):
        with st.spinner(f"Reviewing historical {scanner_category} data..."):
            st.session_state["destiny_scanner_results"] = (
                scan_market_category(scanner_category)
            )
            st.session_state["destiny_scanner_category"] = scanner_category

    scanner_results = st.session_state.get(
        "destiny_scanner_results", pd.DataFrame()
    )

    if isinstance(scanner_results, pd.DataFrame) and not scanner_results.empty:
        analysed_count = int(
            (scanner_results["Data status"] == "Analysed").sum()
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
            data=scanner_results.to_csv(index=False).encode("utf-8"),
            file_name=f"destiny_{scanner_category.lower()}_historical_scan.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Choose a category and tap SCAN SELECTED MARKET.")

    st.caption(
        "Coverage: 1-hour candles, up to 60 days where available. "
        "Data is cached for 5 minutes."
    )
    st.stop()


# =========================================================
# MARKET SELECTION / LOAD
# =========================================================

col1, col2 = st.columns(2)
with col1:
    market_type = st.selectbox("Select market", list(MARKETS.keys()))
with col2:
    asset_name = st.selectbox(
        "Select instrument", list(MARKETS[market_type].keys())
    )

symbol = MARKETS[market_type][asset_name]
chart_timeframe = st.selectbox(
    "Main chart timeframe", TIMEFRAMES, index=2
)

if st.button("REFRESH MARKET DATA", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

try:
    with st.spinner("Loading market data..."):
        frames = load_all_timeframes(symbol)
except Exception as error:
    st.error("The market data could not be loaded.")
    st.code(str(error))
    st.stop()

try:
    with st.spinner("Analysing available timeframes..."):
        analyses = run_all_analyses(frames)
except Exception as error:
    st.error("The analysis process encountered an error.")
    st.code(str(error))
    st.stop()

selected_data = frames.get(chart_timeframe, pd.DataFrame())

with st.spinner(f"Analysing historical S/R on {chart_timeframe}..."):
    sr_selected = run_sr_analysis(selected_data)


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
            st.success(f"{timeframe}: {len(frame)} candles")
        else:
            st.warning(f"{timeframe}: Unavailable")


# =========================================================
# MAIN PRICE CHART
# =========================================================

st.divider()
st.subheader("Price chart")
selected_analysis = analyses.get(chart_timeframe, {})

if (
    not selected_data.empty
    and selected_analysis.get("status") == "ANALYSED"
):
    result = selected_analysis["data"]
    display_chart(
        selected_data, result, sr_selected, symbol, chart_timeframe
    )

    st.metric(
        "Latest completed candle close",
        f"{selected_data['Close'].iloc[-1]:,.5f}"
    )
    st.download_button(
        "Download historical OHLCV data",
        data=selected_data.to_csv().encode("utf-8"),
        file_name=f"{symbol}_{chart_timeframe}.csv",
        mime="text/csv"
    )
elif selected_data.empty:
    st.warning(f"No genuine {chart_timeframe} data is currently available.")
else:
    st.error(selected_analysis.get("error", "Analysis could not be completed."))


# =========================================================
# MULTI-TIMEFRAME CONTEXT
# =========================================================

st.divider()
st.subheader("Destiny AI — MTF Structure Context")

try:
    mtf_context = build_mtf_context(analyses)
    st.caption(mtf_context.get("data_coverage", "MTF coverage unavailable"))

    st.markdown("### Higher Timeframe Context")
    st.info(mtf_context.get(
        "higher_timeframe_summary",
        "Higher timeframe analysis unavailable."
    ))

    st.markdown("### Lower Timeframe Context")
    st.info(mtf_context.get(
        "lower_timeframe_summary",
        "Lower timeframe analysis unavailable."
    ))

    mtf_rows = []
    for timeframe in TIMEFRAMES:
        item = mtf_context.get("timeframes", {}).get(timeframe, {})
        mtf_rows.append({
            "Timeframe": timeframe,
            "Status": item.get("status", "Unavailable"),
            "Bullish BOS": item.get("bullish_bos", 0),
            "Bearish BOS": item.get("bearish_bos", 0),
            "Structure": item.get("structure", "Insufficient data")
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
# SUPPORT / RESISTANCE TABLE
# =========================================================

st.divider()
st.subheader("Historical Support & Resistance Zones")

if isinstance(sr_selected, pd.DataFrame) and not sr_selected.empty:
    st.caption(
        f"Up to 8 S/R zones from the latest 400 {chart_timeframe} candles. "
        "BROKEN means a candle close crossed the outer zone boundary."
    )
    st.dataframe(
        sr_selected.head(8),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No historical S/R zones are available for this timeframe.")


# =========================================================
# MULTI-TIMEFRAME STATISTICS
# =========================================================

st.subheader("Multi-timeframe statistics")
summary = get_statistics(analyses)
st.dataframe(summary, use_container_width=True, hide_index=True)


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
        a.metric("Bullish BOS", stats.get("bullish_bos", 0))
        b.metric("Bearish BOS", stats.get("bearish_bos", 0))
        c.metric("Qualified OB + FVG", stats.get("total_qualified_setups", 0))

        st.write("**Active setups:**", stats.get("active_setups", 0))
        st.write("**Invalidated setups:**", stats.get("invalidated_setups", 0))
        st.write("**Active BSL levels:**", stats.get("active_bsl_levels", 0))
        st.write("**Active SSL levels:**", stats.get("active_ssl_levels", 0))

        setups = result.get("setups", pd.DataFrame())
        if isinstance(setups, pd.DataFrame) and not setups.empty:
            st.write("**Qualified OB + FVG overlap history**")
            st.dataframe(
                setups.tail(30),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(
                "No qualified OB + FVG overlaps were detected in this "
                "historical window. Standalone FVGs are intentionally excluded."
            )

        liquidity = result.get("liquidity", pd.DataFrame())
        if isinstance(liquidity, pd.DataFrame) and not liquidity.empty:
            st.write("**BSL / SSL liquidity history**")
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
    st.write("Market data source: Yahoo Finance through yfinance.")
    st.write("The 4H timeframe is constructed from 1H candles.")
    st.write("The latest potentially incomplete intraday candle is excluded.")
    st.write("Each timeframe is analysed independently.")
    st.write(
        "The strategy engine requires a qualified BOS, displacement, "
        "an opposing order-block candle, and an overlapping FVG."
    )
    st.write(
        "The FVG volume qualification remains enabled in strategy_engine.py. "
        "Markets with missing or zero volume may therefore produce fewer "
        "qualified overlaps."
    )
    st.write(
        "The chart displays only the shared price intersection of qualified "
        "OB and FVG zones. It does not display standalone FVGs."
    )
    st.write(
        "S/R invalidation uses a completed candle close beyond the outer "
        "boundary of the zone. A wick alone does not invalidate it."
    )
    st.write("The strategy analysis uses up to 1,200 candles per timeframe.")
    st.write("The S/R engine analyses up to 400 candles on the selected timeframe.")
    st.write("No fabricated candles are created when market data is unavailable.")

st.caption("DESTINY AI | Historical analysis and virtual paper-testing")
