import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

from strategy_engine import analyze_market
from mtf_context import build_mtf_context
from sr_engine import analyze_sr

st.set_page_config(
    page_title="Destiny AI | Market Details",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TIMEFRAMES = ["1D", "4H", "1H", "15M", "5M", "1M"]
MARKETS = {
    "Crypto": {
        "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD",
        "XRP": "XRP-USD", "BNB": "BNB-USD", "Litecoin": "LTC-USD",
        "Dogecoin": "DOGE-USD",
    },
    "Forex": {
        "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X",
        "GBP/JPY": "GBPJPY=X", "EUR/JPY": "EURJPY=X", "AUD/USD": "AUDUSD=X",
        "USD/CAD": "USDCAD=X", "USD/CHF": "USDCHF=X", "NZD/USD": "NZDUSD=X",
    },
    "Commodities": {
        "Gold": "GC=F", "Silver": "SI=F", "Crude Oil": "CL=F",
        "Brent Oil": "BZ=F", "Natural Gas": "NG=F", "Platinum": "PL=F",
        "Copper": "HG=F",
    },
}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(symbol, interval, period):
    try:
        frame = yf.Ticker(symbol).history(
            period=period, interval=interval, auto_adjust=False,
            actions=False, prepost=False,
        )
        if frame is None or frame.empty:
            return pd.DataFrame()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        needed = ["Open", "High", "Low", "Close"]
        if not all(col in frame.columns for col in needed):
            return pd.DataFrame()
        if "Volume" not in frame.columns:
            frame["Volume"] = 0
        frame = frame[["Open", "High", "Low", "Close", "Volume"]].copy()
        for col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=needed).sort_index()
        return frame[~frame.index.duplicated(keep="last")]
    except Exception:
        return pd.DataFrame()


def completed(frame):
    if frame is None or len(frame) < 2:
        return pd.DataFrame()
    return frame.iloc[:-1].copy()


@st.cache_data(ttl=300, show_spinner=False)
def load_frames(symbol):
    daily = fetch_data(symbol, "1d", "2y")
    if not daily.empty and daily.index[-1].date() >= pd.Timestamp.now(tz="UTC").date():
        daily = daily.iloc[:-1].copy()
    hourly = completed(fetch_data(symbol, "1h", "730d"))
    fifteen = completed(fetch_data(symbol, "15m", "60d"))
    five = completed(fetch_data(symbol, "5m", "60d"))
    one = completed(fetch_data(symbol, "1m", "7d"))
    four = pd.DataFrame()
    if not hourly.empty:
        four = hourly.resample("4h", label="right", closed="right").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna(subset=["Open", "High", "Low", "Close"])
        if len(four) > 1:
            four = four.iloc[:-1].copy()
    return {"1D": daily, "4H": four, "1H": hourly,
            "15M": fifteen, "5M": five, "1M": one}


@st.cache_data(ttl=1800, show_spinner=False)
def analyse_frames(frames):
    output = {}
    for tf in TIMEFRAMES:
        data = frames.get(tf, pd.DataFrame())
        if data.empty:
            output[tf] = {"status": "UNAVAILABLE", "error": "No historical data returned."}
            continue
        try:
            result = analyze_market(data.tail(1200).copy())
            output[tf] = {"status": "ANALYSED", "data": result}
        except Exception as exc:
            output[tf] = {"status": "ERROR", "error": str(exc)}
    return output


def draw_chart(data, result, symbol, timeframe):
    if data.empty:
        st.warning("No candles are available for this timeframe.")
        return
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"],
        low=data["Low"], close=data["Close"], name="Candles",
        increasing_line_color="#00C087", decreasing_line_color="#F04452",
        increasing_fillcolor="#00C087", decreasing_fillcolor="#F04452",
    ))
    visible_start = data.index[-min(300, len(data))]
    visible_end = data.index[-1]
    setups = result.get("setups", pd.DataFrame())
    if isinstance(setups, pd.DataFrame) and not setups.empty:
        for _, item in setups.tail(40).iterrows():
            try:
                t = pd.Timestamp(item["ob_time"])
                if data.index.tz is None and t.tzinfo is not None:
                    t = t.tz_localize(None)
                elif data.index.tz is not None and t.tzinfo is None:
                    t = t.tz_localize(data.index.tz)
                if t < visible_start or t > visible_end:
                    continue
                low, high = float(item["ob_low"]), float(item["ob_high"])
                if low >= high:
                    continue
                bullish = item.get("direction") == "BULLISH"
                fig.add_shape(type="rect", x0=t, x1=visible_end, y0=low, y1=high,
                    fillcolor="rgba(0,192,135,0.13)" if bullish else "rgba(240,68,82,0.13)",
                    line=dict(width=1), layer="below")
                if all(c in setups.columns for c in ["fvg_time", "fvg_low", "fvg_high"]):
                    ft = item.get("fvg_time")
                    if pd.notna(ft):
                        ft = pd.Timestamp(ft)
                        if data.index.tz is None and ft.tzinfo is not None:
                            ft = ft.tz_localize(None)
                        elif data.index.tz is not None and ft.tzinfo is None:
                            ft = ft.tz_localize(data.index.tz)
                        fl, fh = float(item["fvg_low"]), float(item["fvg_high"])
                        if visible_start <= ft <= visible_end and fl < fh:
                            fig.add_shape(type="rect", x0=ft, x1=visible_end, y0=fl, y1=fh,
                                fillcolor="rgba(245,190,55,0.13)",
                                line=dict(color="rgba(245,190,55,0.8)", width=1), layer="below")
            except Exception:
                continue
    liquidity = result.get("liquidity", pd.DataFrame())
    if isinstance(liquidity, pd.DataFrame) and not liquidity.empty:
        if {"swept", "price", "side"}.issubset(liquidity.columns):
            for _, level in liquidity[liquidity["swept"] == False].tail(8).iterrows():
                try:
                    fig.add_hline(y=float(level["price"]), line_dash="dot",
                        line_color="#F04452" if level["side"] == "BSL" else "#00C087",
                        line_width=1)
                except Exception:
                    pass
    fig.update_layout(
        title=f"{symbol} | {timeframe} | Historical Market Details",
        template="plotly_dark", height=500, autosize=True,
        margin=dict(l=8, r=12, t=48, b=30), font=dict(size=13),
        xaxis=dict(type="date", range=[visible_start, visible_end],
            rangeslider=dict(visible=True, thickness=0.08), tickformat="%d %b",
            nticks=6, tickfont=dict(size=11), automargin=True),
        yaxis=dict(title="Price", side="right", autorange=True, automargin=True),
        showlegend=False, dragmode="pan",
    )
    st.plotly_chart(fig, use_container_width=True,
        config={"scrollZoom": True, "displaylogo": False, "responsive": True})
    st.caption("Green areas: bullish OB | Red areas: bearish OB | Yellow areas: FVG | Red dotted lines: BSL | Green dotted lines: SSL")


st.title("DESTINY AI — MARKET DETAILS")
st.caption("Dedicated historical view for one selected market")
st.info("Educational historical analysis and paper-testing only. No live trade signals or transactions.")

c1, c2, c3 = st.columns(3)
with c1:
    market_type = st.selectbox("Market category", list(MARKETS.keys()), key="details_market_type")
with c2:
    asset = st.selectbox("Instrument", list(MARKETS[market_type].keys()), key="details_asset")
with c3:
    timeframe = st.selectbox("Chart timeframe", TIMEFRAMES, index=2, key="details_timeframe")

symbol = MARKETS[market_type][asset]
if st.button("REFRESH THIS MARKET", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

with st.spinner(f"Loading historical data for {asset}..."):
    frames = load_frames(symbol)
with st.spinner("Running historical analysis across available timeframes..."):
    analyses = analyse_frames(frames)

selected = frames.get(timeframe, pd.DataFrame())
selected_item = analyses.get(timeframe, {})
if selected.empty:
    st.error(f"Historical {timeframe} data is unavailable for {asset}.")
    st.stop()

st.subheader(f"{asset} — {timeframe} chart")
if selected_item.get("status") == "ANALYSED":
    result = selected_item["data"]
    draw_chart(selected, result, symbol, timeframe)
    st.metric("Latest completed candle close", f"{selected['Close'].iloc[-1]:,.5f}")
    st.download_button("Download this market's historical data", selected.to_csv().encode("utf-8"),
        file_name=f"{symbol}_{timeframe}_history.csv", mime="text/csv")
else:
    st.warning(selected_item.get("error", "Analysis unavailable."))
    result = {}

st.divider()
st.subheader("Historical support and resistance")
try:
    zones = analyze_sr(selected.tail(400).copy())
    if isinstance(zones, pd.DataFrame) and not zones.empty:
        st.dataframe(zones.head(8), use_container_width=True, hide_index=True)
    else:
        st.info("No S/R zones were found in the available analysis window.")
except Exception as exc:
    st.warning(f"S/R analysis unavailable: {exc}")

st.divider()
st.subheader("Multi-timeframe historical context")
try:
    context = build_mtf_context(analyses)
    st.caption(context.get("data_coverage", "MTF coverage unavailable"))
    st.write("**Higher timeframe context**")
    st.info(context.get("higher_timeframe_summary", "Unavailable"))
    st.write("**Lower timeframe context**")
    st.info(context.get("lower_timeframe_summary", "Unavailable"))
    rows = []
    for tf in TIMEFRAMES:
        item = context.get("timeframes", {}).get(tf, {})
        rows.append({"Timeframe": tf, "Status": item.get("status", "Unavailable"),
            "Bullish BOS": item.get("bullish_bos", 0),
            "Bearish BOS": item.get("bearish_bos", 0),
            "Historical structure": item.get("structure", "Insufficient data")})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
except Exception as exc:
    st.warning(f"MTF context unavailable: {exc}")

st.divider()
st.subheader("Selected timeframe details")
if selected_item.get("status") == "ANALYSED":
    stats = result.get("statistics", {})
    a, b, c, d = st.columns(4)
    a.metric("Bullish BOS", stats.get("bullish_bos", 0))
    b.metric("Bearish BOS", stats.get("bearish_bos", 0))
    c.metric("Qualified OB + FVG", stats.get("total_qualified_setups", 0))
    d.metric("Active setups", stats.get("active_setups", 0))
    setups = result.get("setups", pd.DataFrame())
    if isinstance(setups, pd.DataFrame) and not setups.empty:
        st.write("**Historical qualified OB + FVG setups**")
        st.dataframe(setups.tail(40), use_container_width=True, hide_index=True)
    else:
        st.info("No qualified OB + FVG setups in this analysis window.")
    liquidity = result.get("liquidity", pd.DataFrame())
    if isinstance(liquidity, pd.DataFrame) and not liquidity.empty:
        st.write("**Historical BSL / SSL levels**")
        st.dataframe(liquidity.tail(40), use_container_width=True, hide_index=True)

st.caption("Data: Yahoo Finance via yfinance. 4H candles are aggregated from 1H data. Availability and history depth vary by instrument.")

