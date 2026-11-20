import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(
    page_title="Destiny AI",
    page_icon="📊",
    layout="wide"
)

st.title("DESTINY AI")
st.subheader("Historical Market Data Center")
st.caption("Crypto • Forex • Commodities")

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
        "WTI Crude Oil": "CL=F",
        "Brent Crude Oil": "BZ=F",
        "Natural Gas": "NG=F",
        "Platinum": "PL=F",
        "Copper": "HG=F"
    }
}

@st.cache_data(ttl=300)
def fetch_market_data(symbol, timeframe, period):

    intervals = {
        "1H": "1h",
        "1D": "1d",
        "1W": "1wk"
    }

    interval = "1h" if timeframe == "4H" else intervals[timeframe]

    ticker = yf.Ticker(symbol)

    data = ticker.history(
        period=period,
        interval=interval,
        auto_adjust=False
    )

    if data.empty:
        raise ValueError("No historical data returned for this asset.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    data.index = pd.to_datetime(data.index, utc=True)

    required = ["open", "high", "low", "close"]

    for column in required:
        if column not in data.columns:
            raise ValueError(f"Missing market data: {column}")

    if "volume" not in data.columns:
        data["volume"] = 0

    data = data[required + ["volume"]].dropna(
        subset=required
    )

    if timeframe == "4H":
        data = data.resample("4h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna(subset=required)

    data = data.reset_index()
    data = data.rename(columns={"Datetime": "time", "Date": "time"})

    if "time" not in data.columns:
        data = data.rename(columns={data.columns[0]: "time"})

    return data


st.divider()

left, right = st.columns(2)

with left:
    market = st.selectbox(
        "Select Market",
        list(MARKETS.keys())
    )

    asset = st.selectbox(
        "Select Asset",
        list(MARKETS[market].keys())
    )

with right:
    timeframe = st.selectbox(
        "Select Timeframe",
        ["1H", "4H", "1D", "1W"]
    )

    period = st.selectbox(
        "Historical Period",
        ["1mo", "3mo", "6mo", "1y", "2y"]
    )

symbol = MARKETS[market][asset]

if st.button("LOAD HISTORICAL DATA", type="primary"):

    try:
        with st.spinner("Fetching historical market data..."):

            data = fetch_market_data(
                symbol,
                timeframe,
                period
            )

        st.success("Historical data loaded successfully.")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Latest Close",
            f"{data['close'].iloc[-1]:,.5f}"
        )

        col2.metric(
            "Historical Candles",
            f"{len(data):,}"
        )

        col3.metric(
            "Market",
            market
        )

        st.divider()

        st.subheader(f"{asset} — {timeframe}")

        chart = data.set_index("time")[["close"]]

        st.plotly_chart(go.Figure(data=[go.Candlestick(x=data["time"], open=data["open"], high=data["high"], low=data["low"], close=data["close"])]), use_container_width=True)

        st.subheader("OHLC Historical Data")

        st.dataframe(
            data,
            use_container_width=True,
            hide_index=True
        )

        csv = data.to_csv(index=False)

        st.download_button(
            "DOWNLOAD CSV",
            data=csv,
            file_name=f"Destiny_{asset}_{timeframe}.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error(f"Data loading failed: {error}")
        st.info("Try another asset or historical period.")

st.divider()

st.caption(
    "Destiny AI | Historical market research and educational analysis. "
    "Data is supplied by Yahoo Finance and may differ from broker prices."
)
