
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

from strategy_engine import analyze_market


# =========================================================
# DESTINY AI - MULTI-TIMEFRAME SCALPING ANALYSIS
# =========================================================

st.set_page_config(
    page_title="Destiny AI",
    page_icon="📊",
    layout="wide"
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

        if not all(col in df.columns for col in required):
            return pd.DataFrame()

        if "Volume" not in df.columns:
            df["Volume"] = 0

        df = df[
            ["Open", "High", "Low", "Close", "Volume"]
        ].copy()

        for col in df.columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

        df = df.dropna(
            subset=["Open", "High", "Low", "Close"]
        )

        df = df.sort_index()

        df = df[
            ~df.index.duplicated(keep="last")
        ]

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

        # Exclude the latest potentially incomplete bar.
        if len(result) > 1:
            result = result.iloc[:-1]

        return result

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_all_timeframes(symbol):

    frames = {}

    daily = fetch_market_data(
        symbol, "1d", "2y"
    )

    if not daily.empty:

        today = pd.Timestamp.now(
            tz="UTC"
        ).date()

        if daily.index[-1].date() >= today:
            daily = daily.iloc[:-1].copy()

    hourly = fetch_market_data(
        symbol, "1h", "730d"
    )

    hourly = remove_latest_candle(hourly)

    fifteen = fetch_market_data(
        symbol, "15m", "60d"
    )

    fifteen = remove_latest_candle(fifteen)

    five = fetch_market_data(
        symbol, "5m", "60d"
    )

    five = remove_latest_candle(five)

    one = fetch_market_data(
        symbol, "1m", "7d"
    )

    one = remove_latest_candle(one)

    frames["1D"] = daily
    frames["4H"] = build_four_hour_data(hourly)
    frames["1H"] = hourly
    frames["15M"] = fifteen
    frames["5M"] = five
    frames["1M"] = one

    return frames


# =========================================================
# RUN THE EXISTING ENGINE ON EACH TIMEFRAME
# =========================================================

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
                "error": "No genuine market data available."
            }

            continue

        try:

            result = analyze_market(data)

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
# CHART WITH OB, FVG AND LIQUIDITY OVERLAYS
# =========================================================

def display_chart(data, result, symbol, timeframe):

    chart_data = data.tail(180).copy()

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=chart_data.index,
            open=chart_data["Open"],
            high=chart_data["High"],
            low=chart_data["Low"],
            close=chart_data["Close"],
            name="Price",
            increasing_line_color="#16A36A",
            decreasing_line_color="#E05252",
            increasing_fillcolor="#16A36A",
            decreasing_fillcolor="#E05252",
            whiskerwidth=0.7
        )
    )

    # Draw recent OB and FVG zones.

    setups = result.get("setups", pd.DataFrame())

    if not setups.empty:

        for _, setup in setups.tail(6).iterrows():

            bullish = (
                setup.get("direction") == "BULLISH"
            )

            color = (
                "rgba(22,163,106,0.16)"
                if bullish
                else "rgba(224,82,82,0.16)"
            )

            ob_time = setup.get("ob_time")

            if pd.isna(ob_time):
                continue

            fig.add_shape(
                type="rect",
                x0=ob_time,
                x1=chart_data.index[-1],
                y0=setup["ob_low"],
                y1=setup["ob_high"],
                fillcolor=color,
                line=dict(width=1),
                layer="below"
            )

            fvg_time = setup.get("fvg_time")

            if pd.notna(fvg_time):

                fig.add_shape(
                    type="rect",
                    x0=fvg_time,
                    x1=chart_data.index[-1],
                    y0=setup["fvg_low"],
                    y1=setup["fvg_high"],
                    fillcolor="rgba(230,180,50,0.13)",
                    line=dict(
                        color="rgba(230,180,50,0.7)",
                        width=1
                    ),
                    layer="below"
                )

    # Draw active BSL and SSL levels.

    liquidity = result.get(
        "liquidity",
        pd.DataFrame()
    )

    if not liquidity.empty:

        active = liquidity[
            liquidity["swept"] == False
        ].tail(8)

        for _, level in active.iterrows():

            price = level.get("price")

            if pd.isna(price):
                continue

            side = level.get("side")

            color = (
                "#E05252"
                if side == "BSL"
                else "#16A36A"
            )

            fig.add_hline(
                y=price,
                line_dash="dot",
                line_color=color,
                line_width=1,
                annotation_text=str(side),
                annotation_position="top right"
            )

    fig.update_layout(
        title=f"{symbol} | {timeframe}",
        height=560,
        autosize=True,
        template="plotly_dark",
        margin=dict(
            l=15,
            r=20,
            t=55,
            b=15
        ),
        font=dict(size=14),
        xaxis=dict(
            title="Time",
            rangeslider=dict(visible=False),
            tickfont=dict(size=12),
            showgrid=True,
            gridcolor="rgba(130,130,130,0.15)",
            automargin=True
        ),
        yaxis=dict(
            title="Price",
            side="right",
            tickfont=dict(size=13),
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


# =========================================================
# TIMEFRAME STATISTICS
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

        candles = result["candles"]
        stats = result["statistics"]

        rows.append({
            "Timeframe": timeframe,
            "Status": "ANALYSED",
            "Candles": len(candles),
            "Bullish BOS": stats["bullish_bos"],
            "Bearish BOS": stats["bearish_bos"],
            "Qualified Setups": stats["total_qualified_setups"],
            "Active Setups": stats["active_setups"]
        })

    return pd.DataFrame(rows)


# =========================================================
# APP HEADER
# =========================================================

st.title("DESTINY AI")

st.caption(
    "Multi-timeframe scalping analysis dashboard"
)

st.info(
    "Historical market analysis and paper-testing. "
    "This application does not execute transactions."
)


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

if st.button(
    "REFRESH MARKET DATA",
    use_container_width=True
):

    st.cache_data.clear()
    st.rerun()


# =========================================================
# LOAD ALL DATA
# =========================================================

with st.spinner("Loading market data..."):

    frames = load_all_timeframes(symbol)


# =========================================================
# RUN ANALYSIS
# =========================================================

with st.spinner("Destiny AI is analysing available timeframes..."):

    analyses = run_all_analyses(frames)


# =========================================================
# DATA STATUS
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
# MAIN CHART
# =========================================================

st.divider()

st.subheader("Price chart")

selected_data = frames.get(
    chart_timeframe,
    pd.DataFrame()
)

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
        symbol,
        chart_timeframe
    )

    latest_close = selected_data["Close"].iloc[-1]

    st.metric(
        "Latest completed candle close",
        f"{latest_close:,.5f}"
    )

    st.download_button(
        "Download OHLCV data",
        data=selected_data.to_csv().encode("utf-8"),
        file_name=f"{symbol}_{chart_timeframe}.csv",
        mime="text/csv"
    )

elif selected_data.empty:

    st.warning(
        f"No genuine {chart_timeframe} data is available."
    )

else:

    st.error(
        selected_analysis.get(
            "error",
            "Analysis could not be completed."
        )
    )


# =========================================================
# MULTI-TIMEFRAME DASHBOARD
# =========================================================

st.divider()

st.subheader("Multi-timeframe analysis")

summary = get_statistics(analyses)

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# INDIVIDUAL TIMEFRAME RESULTS
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

        stats = result["statistics"]

        a, b, c = st.columns(3)

        with a:

            st.metric(
                "Bullish BOS",
                stats["bullish_bos"]
            )

        with b:

            st.metric(
                "Bearish BOS",
                stats["bearish_bos"]
            )

        with c:

            st.metric(
                "Qualified setups",
                stats["total_qualified_setups"]
            )

        st.write("**Active setups:**", stats["active_setups"])

        st.write(
            "**Invalidated setups:**",
            stats["invalidated_setups"]
        )

        st.write(
            "**Active BSL levels:**",
            stats["active_bsl_levels"]
        )

        st.write(
            "**Active SSL levels:**",
            stats["active_ssl_levels"]
        )

        setups = result["setups"]

        if not setups.empty:

            st.write("**Qualified OB + FVG setups**")

            st.dataframe(
                setups,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No qualified OB + FVG setups were detected "
                "in this timeframe's available history."
            )

        liquidity = result["liquidity"]

        if not liquidity.empty:

            st.write("**BSL / SSL liquidity history**")

            st.dataframe(
                liquidity.tail(30),
                use_container_width=True,
                hide_index=True
            )


# =========================================================
# DATA LIMITATIONS
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
        "the existing analyze_market() engine."
    )

    st.write(
        "The current engine detects BOS, qualified OB/FVG "
        "overlap, BSL/SSL liquidity and setup invalidation."
    )

    st.write(
        "A combined higher-timeframe decision system and "
        "dedicated support/resistance classifier are not "
        "yet implemented in this engine."
    )

    st.write(
        "Unavailable data is never replaced with fabricated candles."
    )

st.caption(
    "DESTINY AI | Historical market analysis"
)
