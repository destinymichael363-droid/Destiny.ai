
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

from strategy_engine import run_destiny_ai


# =========================================================
# DESTINY AI - SCALPING ANALYSIS DASHBOARD
# Historical market data + multi-timeframe analysis
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


# =========================================================
# DATA FUNCTIONS
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

        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=required)
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

        return df

    except Exception:
        return pd.DataFrame()


def remove_latest_intraday_candle(df):
    """Conservatively exclude the latest possibly incomplete candle."""
    if df is None or df.empty or len(df) < 2:
        return pd.DataFrame()

    return df.iloc[:-1].copy()


def build_four_hour_data(hourly):
    if hourly is None or hourly.empty:
        return pd.DataFrame()

    try:
        four_hour = hourly.resample(
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

        four_hour = four_hour.dropna(
            subset=["Open", "High", "Low", "Close"]
        )

        # Exclude the latest potentially incomplete 4H candle.
        if len(four_hour) > 1:
            four_hour = four_hour.iloc[:-1]

        return four_hour

    except Exception:
        return pd.DataFrame()


def load_all_timeframes(symbol):
    frames = {}

    daily = fetch_market_data(symbol, "1d", "2y")

    if not daily.empty:
        # Exclude today's daily candle if it is still forming.
        today = pd.Timestamp.now(tz="UTC").date()
        if daily.index[-1].date() >= today:
            daily = daily.iloc[:-1].copy()

    hourly = fetch_market_data(symbol, "1h", "730d")
    hourly = remove_latest_intraday_candle(hourly)

    fifteen = fetch_market_data(symbol, "15m", "60d")
    fifteen = remove_latest_intraday_candle(fifteen)

    five = fetch_market_data(symbol, "5m", "60d")
    five = remove_latest_intraday_candle(five)

    one = fetch_market_data(symbol, "1m", "7d")
    one = remove_latest_intraday_candle(one)

    frames["1D"] = daily
    frames["1H"] = hourly
    frames["4H"] = build_four_hour_data(hourly)
    frames["15M"] = fifteen
    frames["5M"] = five
    frames["1M"] = one

    return frames


# =========================================================
# DISPLAY HELPERS
# =========================================================

def get_value(data, keys, default="Not available"):
    if not isinstance(data, dict):
        return default

    for key in keys:
        value = data.get(key)
        if value is not None:
            return value

    return default


def get_timeframe_result(result, timeframe):
    if not isinstance(result, dict):
        return {}

    possible_sections = [
        result.get("timeframes"),
        result.get("timeframe_analysis"),
        result.get("analyses"),
    ]

    for section in possible_sections:
        if isinstance(section, dict):
            if timeframe in section:
                return section[timeframe]

    return {}


def display_candlestick_chart(df, symbol, timeframe):
    chart_data = df.tail(180).copy()

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

    fig.update_layout(
        title=f"{symbol}  |  {timeframe}",
        height=560,
        autosize=True,
        template="plotly_dark",
        margin=dict(l=15, r=15, t=55, b=15),
        font=dict(size=14),
        xaxis=dict(
            title="Time",
            showgrid=True,
            gridcolor="rgba(130,130,130,0.15)",
            rangeslider=dict(visible=False),
            tickfont=dict(size=12),
            automargin=True
        ),
        yaxis=dict(
            title="Price",
            side="right",
            showgrid=True,
            gridcolor="rgba(130,130,130,0.15)",
            tickfont=dict(size=13),
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
# APP HEADER
# =========================================================

st.title("DESTINY AI")
st.caption("Multi-timeframe scalping analysis dashboard")

st.info(
    "Historical market analysis and paper-testing only. "
    "Signals are educational analysis, not guaranteed outcomes."
)


# =========================================================
# MARKET SELECTION
# =========================================================

left, right = st.columns(2)

with left:
    market_type = st.selectbox(
        "Market",
        list(MARKETS.keys())
    )

with right:
    asset_name = st.selectbox(
        "Instrument",
        list(MARKETS[market_type].keys())
    )

symbol = MARKETS[market_type][asset_name]

chart_timeframe = st.selectbox(
    "Main chart timeframe",
    TIMEFRAMES,
    index=2
)

if st.button("REFRESH MARKET DATA", use_container_width=True):
    st.cache_data.clear()
    st.rerun()


# =========================================================
# LOAD DATA
# =========================================================

with st.spinner("Loading historical market data..."):
    frames = load_all_timeframes(symbol)

available = [
    tf for tf in TIMEFRAMES
    if tf in frames and not frames[tf].empty
]

st.subheader("Market data status")

status_columns = st.columns(3)

for index, timeframe in enumerate(TIMEFRAMES):
    column = status_columns[index % 3]
    data = frames.get(timeframe, pd.DataFrame())

    with column:
        if not data.empty:
            st.success(
                f"{timeframe}: Available "
                f"({len(data)} candles)"
            )
        else:
            st.warning(f"{timeframe}: Unavailable")


# =========================================================
# MAIN PRICE CHART
# =========================================================

st.divider()
st.subheader("Price chart")

selected_data = frames.get(
    chart_timeframe,
    pd.DataFrame()
)

if not selected_data.empty:
    display_candlestick_chart(
        selected_data,
        symbol,
        chart_timeframe
    )

    latest_close = selected_data["Close"].iloc[-1]

    st.metric(
        "Latest completed candle close",
        f"{latest_close:,.5f}"
    )

    csv_data = selected_data.to_csv().encode("utf-8")

    st.download_button(
        "Download chart data (CSV)",
        data=csv_data,
        file_name=f"{symbol}_{chart_timeframe}.csv",
        mime="text/csv"
    )

else:
    st.warning(
        f"No genuine {chart_timeframe} data is currently "
        "available for this instrument."
    )


# =========================================================
# DESTINY AI MULTI-TIMEFRAME ENGINE
# =========================================================

st.divider()
st.subheader("Destiny AI analysis")

if len(available) == 6:

    with st.spinner("Analysing all six timeframes..."):
        try:
            analysis = run_destiny_ai(frames)

            if isinstance(analysis, dict):

                decision = get_value(
                    analysis,
                    ["decision", "overall_decision", "status"]
                )

                direction = get_value(
                    analysis,
                    ["direction", "bias", "market_direction"]
                )

                reason = get_value(
                    analysis,
                    ["reason", "explanation", "summary"]
                )

                st.markdown("### Overall assessment")

                a, b = st.columns(2)

                with a:
                    st.metric("Decision", str(decision))

                with b:
                    st.metric("Market direction", str(direction))

                st.write("**Analysis explanation:**")
                st.write(reason)

                st.markdown("### Timeframe breakdown")

                for timeframe in TIMEFRAMES:
                    details = get_timeframe_result(
                        analysis,
                        timeframe
                    )

                    with st.expander(
                        f"{timeframe} analysis",
                        expanded=(timeframe == "1H")
                    ):

                        if isinstance(details, dict) and details:
                            st.write(
                                "**Direction:**",
                                get_value(
                                    details,
                                    ["direction", "bias", "trend"]
                                )
                            )

                            st.write(
                                "**Structure:**",
                                get_value(
                                    details,
                                    ["structure", "market_structure"]
                                )
                            )

                            st.write(
                                "**Decision:**",
                                get_value(
                                    details,
                                    ["decision", "status"]
                                )
                            )

                            st.write(
                                "**Reason:**",
                                get_value(
                                    details,
                                    ["reason", "explanation", "summary"]
                                )
                            )

                        else:
                            st.write(
                                "The engine did not return a separate "
                                "breakdown for this timeframe."
                            )

                with st.expander("Full engine output"):
                    st.json(analysis)

            else:
                st.write(analysis)

        except Exception as error:
            st.error(
                "The strategy engine returned an error. "
                "Check that strategy_engine.py is saved correctly."
            )
            st.code(str(error))

else:
    missing = [
        tf for tf in TIMEFRAMES
        if frames.get(tf, pd.DataFrame()).empty
    ]

    st.warning(
        "Destiny AI needs all six genuine timeframes before "
        "it can produce a complete multi-timeframe assessment."
    )

    st.write("Unavailable timeframes:", ", ".join(missing))

    st.info(
        "The app will not manufacture missing candles or "
        "pretend that a different timeframe is equivalent."
    )


# =========================================================
# DATA NOTES
# =========================================================

st.divider()

with st.expander("Data and analysis limitations"):
    st.write(
        "- Data is retrieved from Yahoo Finance through yfinance."
    )
    st.write(
        "- Intraday data availability depends on the instrument "
        "and the data provider."
    )
    st.write(
        "- The 4H chart is constructed from 1H candles."
    )
    st.write(
        "- The latest potentially incomplete intraday candles "
        "are excluded."
    )
    st.write(
        "- Historical data may be delayed, incomplete or revised."
    )
    st.write(
        "- This application does not connect to a broker or "
        "execute transactions."
    )

st.caption("Destiny AI | Historical analysis and paper-testing")
