from __future__ import annotations
from setup_bridge import register_setups
from datetime import datetime, timezone
import urllib.request

import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

from mtf_context import build_mtf_context
from strategy_engine import analyze_market
from sr_engine import analyze_sr
from auth_guard import require_login

# =========================================================
# DESTINY AI — HISTORICAL ANALYSIS / EDUCATIONAL REVIEW
# Historical data only. No broker or live-order connection.
# =========================================================

st.set_page_config(
    page_title="Destiny AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)
require_login()


TIMEFRAMES = ["1D", "4H", "1H", "15M", "5M", "1M"]
SR_TIMEFRAMES = ["4H", "1H", "15M"]
SR_CLASSIFICATION = {"4H": "MAJOR", "1H": "MAJOR", "15M": "MINOR"}

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

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]



# =========================================================
# OPTIONAL NOTIFICATION
# =========================================================

def send_app_notification(title: str, message: str) -> bool:
    """Send an optional ntfy notification using Streamlit Secrets."""
    try:
        topic = str(st.secrets.get("NTFY_TOPIC", "")).strip()
        if not topic:
            return False
        request = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=str(message).encode("utf-8"),
            headers={"Title": str(title)},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status in (200, 201)
    except Exception:
        return False


if st.sidebar.button("Test Android Notification", key="destiny_test_ntfy"):
    if send_app_notification("Destiny AI", "Destiny AI notification test completed."):
        st.sidebar.success("Test notification sent.")
    else:
        st.sidebar.warning("Notification unavailable. Check NTFY_TOPIC in Secrets.")


# =========================================================
# HISTORICAL DATA HELPERS
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Fetch and normalize historical OHLCV; return empty frame on provider errors."""
    try:
        frame = yf.Ticker(symbol).history(
            period=period,
            interval=interval,
            auto_adjust=False,
            actions=False,
            prepost=False,
        )
        if frame is None or frame.empty:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        if not all(c in frame.columns for c in ["Open", "High", "Low", "Close"]):
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        if "Volume" not in frame.columns:
            frame["Volume"] = 0
        frame = frame[OHLCV_COLUMNS].copy()
        for column in OHLCV_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
        frame = frame.sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        # Reject malformed OHLC rows rather than passing them to engines.
        valid = (
            (frame["High"] >= frame[["Open", "Close", "Low"]].max(axis=1))
            & (frame["Low"] <= frame[["Open", "Close", "High"]].min(axis=1))
        )
        frame = frame.loc[valid]
        return frame
    except Exception:
        return pd.DataFrame(columns=OHLCV_COLUMNS)



def remove_latest_candle(frame: pd.DataFrame) -> pd.DataFrame:
    """Conservatively omit the newest candle, which may still be forming."""
    if frame is None or frame.empty or len(frame) < 2:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    return frame.iloc[:-1].copy()



def build_four_hour_data(hourly: pd.DataFrame) -> pd.DataFrame:
    """Build 4H candles from hourly data and omit the newest resampled bucket."""
    if hourly is None or hourly.empty or len(hourly) < 5:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    try:
        resampled = hourly.resample("4h", label="right", closed="right").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        )
        resampled = resampled.dropna(subset=["Open", "High", "Low", "Close"])
        resampled = resampled[OHLCV_COLUMNS]
        # The final bucket can be partial even if the source's last hourly bar was removed.
        return resampled.iloc[:-1].copy() if len(resampled) > 1 else pd.DataFrame(columns=OHLCV_COLUMNS)
    except Exception:
        return pd.DataFrame(columns=OHLCV_COLUMNS)



@st.cache_data(ttl=300, show_spinner=False)
def load_all_timeframes(symbol: str) -> dict[str, pd.DataFrame]:
    """Load each source interval; data is cached briefly to reduce provider calls."""
    daily_raw = fetch_market_data(symbol, "1d", "2y")
    daily = remove_latest_candle(daily_raw)

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
        "1M": one,
    }


# =========================================================
# ANALYSIS ENGINES
# =========================================================

@st.cache_data(ttl=1800, show_spinner=False)
def run_all_analyses(frames: dict[str, pd.DataFrame]) -> dict:
    results = {}
    for timeframe in TIMEFRAMES:
        data = frames.get(timeframe, pd.DataFrame())
        if data is None or data.empty:
            results[timeframe] = {"status": "UNAVAILABLE", "error": "Historical data unavailable."}
            continue
        try:
            output = analyze_market(data.tail(1200).copy())
            if not isinstance(output, dict):
                raise TypeError("Analysis engine returned an unexpected result type.")
            results[timeframe] = {"status": "ANALYSED", "data": output, "error": None}
        except Exception as exc:
            results[timeframe] = {"status": "ERROR", "error": str(exc)}
    return results



@st.cache_data(ttl=1800, show_spinner=False)
def run_sr_analysis(data: pd.DataFrame, source_timeframe: str) -> pd.DataFrame:
    """Run S/R independently on a cleaned source-timeframe dataset."""
    if data is None or data.empty:
        return pd.DataFrame()
    clean = data.copy().sort_index()
    clean = clean[~clean.index.duplicated(keep="last")]
    required = ["High", "Low", "Close"]
    if not all(col in clean.columns for col in required):
        raise ValueError(f"{source_timeframe}: required OHLC columns are missing.")
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
        max_zones=40,
    )
    if not isinstance(zones, pd.DataFrame):
        raise TypeError(f"{source_timeframe}: S/R engine returned a non-DataFrame result.")
    if zones.empty:
        return pd.DataFrame()
    zones = zones.copy()
    if "Confirmed Time" not in zones.columns:
        raise ValueError(f"{source_timeframe}: S/R output has no Confirmed Time column.")
    zones["Source Timeframe"] = source_timeframe
    zones["Classification"] = SR_CLASSIFICATION.get(source_timeframe, "UNCLASSIFIED")
    return zones.sort_values("Confirmed Time", ascending=False).head(8).reset_index(drop=True)



@st.cache_data(ttl=1800, show_spinner=False)
def build_combined_sr(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    layers = []
    for timeframe in SR_TIMEFRAMES:
        source = frames.get(timeframe, pd.DataFrame())
        layer = run_sr_analysis(source, timeframe)
        if isinstance(layer, pd.DataFrame) and not layer.empty:
            layers.append(layer)
    return pd.concat(layers, ignore_index=True) if layers else pd.DataFrame()


# =========================================================
# TIME ALIGNMENT / PLOT HELPERS
# =========================================================

def align_timestamp(value, reference) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    ref_timestamp = pd.Timestamp(reference)
    if ref_timestamp.tzinfo is None and timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    elif ref_timestamp.tzinfo is not None and timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(ref_timestamp.tzinfo)
    return timestamp



def draw_sr_zones(fig, zones, data, start_time, end_time) -> None:
    required = ["Side", "Zone Low", "Zone High"]
    if not isinstance(zones, pd.DataFrame) or zones.empty or not all(c in zones.columns for c in required):
        return
    try:
        latest_close = float(data["Close"].iloc[-1])
    except Exception:
        return
    parsed = []
    for _, zone in zones.iterrows():
        try:
            side = str(zone["Side"]).upper()
            low, high = float(zone["Zone Low"]), float(zone["Zone High"])
            source_tf = str(zone.get("Source Timeframe", "S/R"))
            classification = str(zone.get("Classification", "UNCLASSIFIED"))
            if side not in ("SUPPORT", "RESISTANCE") or not (pd.notna(low) and pd.notna(high)) or low >= high:
                continue
            parsed.append((abs((low + high) / 2 - latest_close), zone, side, low, high, source_tf, classification))
        except Exception:
            continue
    if not parsed:
        return
    # To reduce mobile clutter, display the nearest 8 zones while retaining each S/R layer.
    by_layer = {}
    for item in sorted(parsed, key=lambda row: row[0]):
        by_layer.setdefault(item[5], [])
        if len(by_layer[item[5]]) < 3:
            by_layer[item[5]].append(item)
    selected = [item for layer_items in by_layer.values() for item in layer_items]
    selected.sort(key=lambda row: row[0])
    label_keys = {(item[5], item[2], item[3], item[4], str(item[1].get("Confirmed Time", ""))) for item in selected}
    label_offset = 0
    for _, zone, side, low, high, source_tf, classification in selected:
        try:
            zone_time = zone.get("Confirmed Time", zone.get("Pivot Time", data.index[0]))
            if pd.isna(zone_time):
                zone_time = data.index[0]
            zone_start = max(align_timestamp(zone_time, start_time), start_time)
            if zone_start > end_time:
                continue
            color = "#00C087" if side == "SUPPORT" else "#F04452"
            dash = {"4H": "solid", "1H": "dash", "15M": "dot"}.get(source_tf, "solid")
            alpha = {"4H": 0.105, "1H": 0.075, "15M": 0.045}.get(source_tf, 0.06)
            fill = f"rgba(0,192,135,{alpha})" if side == "SUPPORT" else f"rgba(240,68,82,{alpha})"
            mid = (low + high) / 2
            fig.add_shape(type="rect", x0=zone_start, x1=end_time, y0=low, y1=high,
                         fillcolor=fill, line=dict(color=color, width=1.25 if source_tf == "4H" else 1, dash=dash), layer="below")
            fig.add_shape(type="line", x0=zone_start, x1=end_time, y0=mid, y1=mid,
                         line=dict(color=color, width=0.8, dash=dash), layer="below")
            key = (source_tf, side, low, high, str(zone.get("Confirmed Time", "")))
            if key in label_keys:
                fig.add_annotation(x=end_time, y=mid, text=f"{source_tf} {classification} · {side}",
                                 showarrow=False, xanchor="right", yanchor="middle", yshift=(label_offset % 3 - 1) * 12,
                                 font=dict(size=8, color=color), bgcolor="rgba(11,14,17,0.82)", borderpad=2)
                label_offset += 1
        except Exception:
            continue



def draw_ob_fvg_overlaps(fig, setups, data, start_time, end_time) -> int:
    required = ["ob_time", "ob_low", "ob_high", "fvg_time", "fvg_low", "fvg_high", "direction"]
    if not isinstance(setups, pd.DataFrame) or setups.empty or not all(c in setups.columns for c in required):
        return 0
    drawn = 0
    active_label_count = 0
    for _, setup in setups.tail(100).iterrows():
        if drawn >= 5:
            break
        try:
            ob_time, fvg_time = setup["ob_time"], setup["fvg_time"]
            if pd.isna(ob_time) or pd.isna(fvg_time):
                continue
            overlap_time = max(align_timestamp(ob_time, start_time), align_timestamp(fvg_time, start_time))
            if overlap_time > end_time:
                continue
            ob_low, ob_high = float(setup["ob_low"]), float(setup["ob_high"])
            fvg_low, fvg_high = float(setup["fvg_low"]), float(setup["fvg_high"])
            low, high = max(ob_low, fvg_low), min(ob_high, fvg_high)
            if low >= high:
                continue
            direction = str(setup["direction"]).upper()
            if direction == "BULLISH":
                color = "#00C087"
                invalid_mask = (data.index >= overlap_time) & (data["Close"] < max(ob_low, fvg_low))
            elif direction == "BEARISH":
                color = "#F04452"
                invalid_mask = (data.index >= overlap_time) & (data["Close"] > min(ob_high, fvg_high))
            else:
                continue
            invalid_rows = data.loc[invalid_mask]
            setup_end = end_time
            invalidated = not invalid_rows.empty
            if invalidated:
                invalid_time = align_timestamp(invalid_rows.index[0], start_time)
                if invalid_time < start_time:
                    continue
                setup_end = min(invalid_time, end_time)
            elif str(setup.get("status", "ACTIVE")).upper() == "INVALIDATED":
                # Engine says invalidated but its timestamp isn't in the visible data.
                continue
            visible_start = max(overlap_time, start_time)
            if visible_start > setup_end:
                continue
            fill = ("rgba(240,68,82,0.035)" if direction == "BEARISH" else "rgba(0,192,135,0.035)") if invalidated else ("rgba(240,68,82,0.12)" if direction == "BEARISH" else "rgba(0,192,135,0.12)")
            fig.add_shape(type="rect", x0=visible_start, x1=setup_end, y0=low, y1=high, fillcolor=fill,
                         line=dict(color=color, width=0.7 if invalidated else 1, dash="dot" if invalidated else "solid"), layer="below")
            if not invalidated and active_label_count < 2:
                label = "BULL OB/FVG" if direction == "BULLISH" else "BEAR OB/FVG"
                fig.add_annotation(x=end_time, y=(low + high) / 2, text=label, showarrow=False,
                                 xanchor="right", yanchor="middle", yshift=active_label_count * 12,
                                 font=dict(size=8, color=color), bgcolor="rgba(11,14,17,0.82)")
                active_label_count += 1
            drawn += 1
        except Exception:
            continue
    return drawn


# =========================================================
# MOBILE CHART
# =========================================================

def display_chart(data, result, sr_zones, symbol, timeframe):
    chart_data = data.tail(1200).copy()
    if chart_data.empty:
        st.warning("No candles are available.")
        return
    left, right = st.columns(2)
    with left:
        visible_count = st.slider("Candles on screen", 30, 180, 60, 10, key=f"visible_{symbol}_{timeframe}")
    with right:
        vertical_scale = st.slider("Vertical zoom", 0.6, 2.0, 1.0, 0.1, key=f"vertical_{symbol}_{timeframe}",
                                 help="Adjust the visible vertical price range.")
    start_time, end_time = chart_data.index[0], chart_data.index[-1]
    latest_close = float(chart_data["Close"].iloc[-1])
    visible = chart_data.tail(visible_count)
    first_visible = visible.index[0]
    visible_low, visible_high = float(visible["Low"].min()), float(visible["High"].max())
    span = max(visible_high - visible_low, abs(latest_close) * 0.002)
    center = (visible_high + visible_low) / 2
    half_range = max(span * 0.55 * vertical_scale, abs(latest_close) * 0.0005)
    y_range = [center - half_range, center + half_range]

    fig = go.Figure()
    volume_custom = chart_data["Volume"].fillna(0).to_numpy().reshape(-1, 1)
    fig.add_trace(go.Candlestick(
        x=chart_data.index, open=chart_data["Open"], high=chart_data["High"], low=chart_data["Low"], close=chart_data["Close"],
        customdata=volume_custom, name="Price", increasing_line_color="#20C997", decreasing_line_color="#F05B65",
        increasing_fillcolor="#20C997", decreasing_fillcolor="#F05B65", whiskerwidth=1,
        hovertemplate=("<b>%{x|%d %b %Y %H:%M}</b><br>Open: %{open:,.5f}<br>High: %{high:,.5f}<br>"
                     "Low: %{low:,.5f}<br>Close: %{close:,.5f}<br>Volume: %{customdata[0]:,.0f}<extra></extra>"),
    ))
    draw_sr_zones(fig, sr_zones, chart_data, start_time, end_time)
    overlap_count = draw_ob_fvg_overlaps(fig, result.get("setups", pd.DataFrame()), chart_data, start_time, end_time)

    liquidity = result.get("liquidity", pd.DataFrame())
    if isinstance(liquidity, pd.DataFrame) and not liquidity.empty and {"swept", "price", "side"}.issubset(liquidity.columns):
        active_liquidity = liquidity[liquidity["swept"].eq(False)].tail(6)
        for _, level in active_liquidity.iterrows():
            try:
                price = float(level["price"])
                color = "#F05B65" if str(level["side"]).upper() == "BSL" else "#20C997"
                fig.add_hline(y=price, line_dash="dot", line_color=color, line_width=1.2)
            except Exception:
                continue

    fig.add_hline(y=latest_close, line_color="#FFD54F", line_width=1.7,
                  annotation_text=f"Latest close {latest_close:,.5f}", annotation_position="top right",
                  annotation_font=dict(color="#FFD54F", size=10), annotation_bgcolor="rgba(20,20,20,0.88)")
    reset_counter = st.session_state.get("destiny_chart_reset_counter", 0)
    fig.update_layout(
        title=dict(text=f"{symbol} | {timeframe}", x=0.02, xanchor="left", font=dict(size=18, color="#E6EAF0")),
        template="plotly_dark", paper_bgcolor="#0B0E11", plot_bgcolor="#0B0E11", height=640, autosize=True,
        margin=dict(l=8, r=82, t=58, b=35), font=dict(family="Arial, sans-serif", size=13, color="#D7DCE2"),
        hoverlabel=dict(bgcolor="#20262D", bordercolor="#59636E", font=dict(size=12, color="#FFFFFF")),
        hovermode="closest", dragmode="pan", showlegend=False,
        uirevision=f"destiny-{symbol}-{timeframe}-{reset_counter}",
        xaxis=dict(type="date", range=[first_visible, end_time], rangeslider=dict(visible=False), tickformat="%d %b", nticks=5,
                   showgrid=True, gridcolor="rgba(180,190,200,0.10)", showspikes=True, spikemode="across", spikesnap="cursor", automargin=True),
        yaxis=dict(side="right", range=y_range, autorange=False, fixedrange=False, showgrid=True,
                   gridcolor="rgba(180,190,200,0.10)", showspikes=True, spikemode="across", spikesnap="cursor", automargin=True),
    )
    st.plotly_chart(fig, use_container_width=True, config={
        "scrollZoom": True, "displayModeBar": True, "displaylogo": False, "responsive": True,
        "doubleClick": "reset", "modeBarButtonsToAdd": ["pan2d", "zoom2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
    }, key=f"destiny-chart-{symbol}-{timeframe}")
    st.caption("4H major S/R: solid | 1H major: dashed | 15M minor: dotted. Green: support/bullish; red: resistance/bearish. Yellow: latest completed close. S/R outer-boundary invalidation is close-based on its source timeframe.")
    if overlap_count == 0:
        st.info("No qualified OB/FVG overlap is visible in this chart window. Standalone FVGs are not drawn.")


# =========================================================
# STATISTICS / HISTORICAL SCANNER
# =========================================================

def get_statistics(results: dict) -> pd.DataFrame:
    rows = []
    for timeframe in TIMEFRAMES:
        item = results.get(timeframe, {})
        if item.get("status") != "ANALYSED":
            rows.append({"Timeframe": timeframe, "Status": item.get("status", "UNKNOWN"), "Candles": 0,
                        "Bullish BOS": 0, "Bearish BOS": 0, "Qualified Setups": 0, "Active Setups": 0})
            continue
        result = item["data"]
        stats = result.get("statistics", {})
        rows.append({"Timeframe": timeframe, "Status": "ANALYSED", "Candles": len(result.get("candles", [])),
                    "Bullish BOS": stats.get("bullish_bos", 0), "Bearish BOS": stats.get("bearish_bos", 0),
                    "Qualified Setups": stats.get("total_qualified_setups", 0), "Active Setups": stats.get("active_setups", 0)})
    return pd.DataFrame(rows)



@st.cache_data(ttl=300, show_spinner=False)
def scan_market_category(category: str) -> pd.DataFrame:
    rows = []
    for instrument, ticker in MARKETS.get(category, {}).items():
        try:
            history = remove_latest_candle(fetch_market_data(ticker, "1h", "60d"))
            if history.empty or len(history) < 40:
                rows.append({"Instrument": instrument, "Symbol": ticker, "Data status": "Unavailable / insufficient",
                            "Candles analysed": len(history), "Bullish BOS history": "—", "Bearish BOS history": "—",
                            "Historical qualified setups": "—", "Active historical setups": "—"})
                continue
            analysis = analyze_market(history.tail(1200).copy())
            stats = analysis.get("statistics", {}) if isinstance(analysis, dict) else {}
            rows.append({"Instrument": instrument, "Symbol": ticker, "Data status": "Analysed",
                        "Candles analysed": len(analysis.get("candles", [])),
                        "Bullish BOS history": int(stats.get("bullish_bos", 0) or 0),
                        "Bearish BOS history": int(stats.get("bearish_bos", 0) or 0),
                        "Historical qualified setups": int(stats.get("total_qualified_setups", 0) or 0),
                        "Active historical setups": int(stats.get("active_setups", 0) or 0)})
        except Exception:
            rows.append({"Instrument": instrument, "Symbol": ticker, "Data status": "Analysis unavailable",
                        "Candles analysed": 0, "Bullish BOS history": "—", "Bearish BOS history": "—",
                        "Historical qualified setups": "—", "Active historical setups": "—"})
    return pd.DataFrame(rows)


# =========================================================
# APP UI
# =========================================================

st.title("DESTINY AI")
st.caption("Multi-timeframe historical market analysis")
st.info("Educational historical analysis only. No broker connection, live orders, or trade execution.")
workspace = st.radio("Choose workspace", ["Main Analysis", "Historical Market Scanner"], horizontal=True, key="destiny_workspace")


# Scanner workspace
if workspace == "Historical Market Scanner":
    st.divider()
    st.subheader("Historical Market Scanner")
    st.caption("Reviews completed historical 1-hour candles. Results are descriptive, not forecasts.")
    category = st.selectbox("Market category", list(MARKETS.keys()), key="scanner_market_category")
    left, right = st.columns(2)
    with left:
        run_scan = st.button("SCAN SELECTED MARKET", use_container_width=True, type="primary")
    with right:
        refresh_scan = st.button("REFRESH SCANNER DATA", use_container_width=True)
    if refresh_scan:
        scan_market_category.clear()
        fetch_market_data.clear()
        st.session_state.pop("destiny_scanner_results", None)
        st.rerun()
    if run_scan or "destiny_scanner_results" not in st.session_state or st.session_state.get("destiny_scanner_category") != category:
        with st.spinner(f"Reviewing historical {category} data..."):
            st.session_state["destiny_scanner_results"] = scan_market_category(category)
            st.session_state["destiny_scanner_category"] = category
    scan_results = st.session_state.get("destiny_scanner_results", pd.DataFrame())
    if isinstance(scan_results, pd.DataFrame) and not scan_results.empty:
        analysed_count = int(scan_results["Data status"].eq("Analysed").sum())
        st.metric("Instruments analysed", f"{analysed_count} / {len(scan_results)}")
        st.dataframe(scan_results, use_container_width=True, hide_index=True)
        st.download_button("Download scanner results", data=scan_results.to_csv(index=False).encode("utf-8"),
                         file_name=f"destiny_{category.lower()}_historical_scan.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("Choose a category and tap SCAN SELECTED MARKET.")
    st.caption("Coverage: 1-hour candles, up to 60 days where available.")
    st.stop()


# Market selection
col1, col2 = st.columns(2)
with col1:
    market_type = st.selectbox("Select market", list(MARKETS.keys()), key="destiny_market_type")
with col2:
    asset_name = st.selectbox("Select instrument", list(MARKETS[market_type].keys()), key="destiny_asset_name")
symbol = MARKETS[market_type][asset_name]
chart_timeframe = st.selectbox("Main chart timeframe", TIMEFRAMES, index=2, key="destiny_chart_timeframe")

if st.button("REFRESH MARKET DATA", use_container_width=True, key="destiny_refresh_market"):
    fetch_market_data.clear()
    load_all_timeframes.clear()
    run_all_analyses.clear()
    run_sr_analysis.clear()
    build_combined_sr.clear()
    st.rerun()

try:
    with st.spinner("Loading historical market data..."):
        frames = load_all_timeframes(symbol)
except Exception as exc:
    st.error("Market data could not be loaded.")
    st.caption(str(exc))
    st.stop()

try:
    with st.spinner("Analysing available timeframes..."):
        analyses = run_all_analyses(frames)
except Exception as exc:
    st.error("Analysis process encountered an error.")
    st.caption(str(exc))
    st.stop()

selected_data = frames.get(chart_timeframe, pd.DataFrame())
with st.spinner("Building 4H major, 1H major and 15M minor historical S/R..."):
    try:
        sr_all = build_combined_sr(frames)
        sr_error = None
    except Exception as exc:
        sr_all, sr_error = pd.DataFrame(), str(exc)

# Data availability
st.divider()
st.subheader("Market data availability")
status_columns = st.columns(3)
for index, timeframe in enumerate(TIMEFRAMES):
    frame = frames.get(timeframe, pd.DataFrame())
    with status_columns[index % 3]:
        if frame is not None and not frame.empty:
            st.success(f"{timeframe}: {len(frame)} candles")
        else:
            st.warning(f"{timeframe}: Unavailable")

# Chart
st.divider()
st.subheader("Price chart")
st.caption("Starts with the latest 60 candles. Review the loaded history using chart pan/zoom. S/R layers remain independent of the selected chart timeframe.")
if st.button("RETURN TO LATEST", use_container_width=True, key="destiny_return_latest"):
    st.session_state["destiny_chart_reset_counter"] = st.session_state.get("destiny_chart_reset_counter", 0) + 1
    st.rerun()
selected_analysis = analyses.get(chart_timeframe, {})
if not selected_data.empty and selected_analysis.get("status") == "ANALYSED":
    display_chart(selected_data, selected_analysis["data"], sr_all, symbol, chart_timeframe)
    st.metric("Latest completed candle close", f"{float(selected_data['Close'].iloc[-1]):,.5f}")
    st.download_button("Download historical OHLCV data", data=selected_data.to_csv().encode("utf-8"),
                     file_name=f"{symbol}_{chart_timeframe}.csv", mime="text/csv")
elif selected_data.empty:
    st.warning(f"No genuine {chart_timeframe} historical data is currently available.")
else:
    st.error(selected_analysis.get("error", "Analysis could not be completed."))

# MTF context
st.divider()
st.subheader("Destiny AI — MTF Structure Context")
try:
    mtf_context = build_mtf_context(analyses)
    st.caption(mtf_context.get("data_coverage", "MTF coverage unavailable"))
    st.markdown("### Higher Timeframe Context")
    st.info(mtf_context.get("higher_timeframe_summary", "Higher timeframe analysis unavailable."))
    st.markdown("### Lower Timeframe Context")
    st.info(mtf_context.get("lower_timeframe_summary", "Lower timeframe analysis unavailable."))
    mtf_rows = []
    for timeframe in TIMEFRAMES:
        item = mtf_context.get("timeframes", {}).get(timeframe, {})
        mtf_rows.append({"Timeframe": timeframe, "Status": item.get("status", "Unavailable"),
                         "Bullish BOS": item.get("bullish_bos", 0), "Bearish BOS": item.get("bearish_bos", 0),
                         "Structure": item.get("structure", "Insufficient data")})
    st.dataframe(pd.DataFrame(mtf_rows), use_container_width=True, hide_index=True)
    st.caption("MTF context summarizes historical structure and does not forecast price direction.")
except Exception as exc:
    st.error("MTF context could not be generated.")
    st.caption(str(exc))

# S/R table
st.divider()
st.subheader("Historical Support & Resistance Zones")
if sr_error:
    st.error("S/R calculation encountered an error.")
    st.code(sr_error)
if isinstance(sr_all, pd.DataFrame) and not sr_all.empty:
    st.caption("Combined S/R: 4H major, 1H major and 15M minor. Each layer is calculated from its own completed-candle history.")
    display_columns = [c for c in ["Source Timeframe", "Classification", "Side", "Zone Low", "Zone High", "Midline",
                                  "Confirmed Time", "Pivot Time", "Rejections", "Strength", "Strong"] if c in sr_all.columns]
    st.dataframe(sr_all[display_columns], use_container_width=True, hide_index=True)
else:
    st.info("No 4H, 1H or 15M historical S/R zones are currently available. Check data availability and any S/R error above.")

# Statistics
st.subheader("Multi-timeframe statistics")
st.dataframe(get_statistics(analyses), use_container_width=True, hide_index=True)


# Detailed analysis tables
st.subheader("Detailed timeframe results")
for timeframe in TIMEFRAMES:
    item = analyses.get(timeframe, {})
    with st.expander(f"{timeframe} analysis", expanded=(timeframe == "1H")):
        if item.get("status") == "UNAVAILABLE":
            st.warning(item.get("error", "Data unavailable."))
            continue
        if item.get("status") == "ERROR":
            st.error(item.get("error", "Analysis error."))
            continue
        result = item["data"]
        stats = result.get("statistics", {})
        a, b, c = st.columns(3)
        a.metric("Bullish BOS history", stats.get("bullish_bos", 0))
        b.metric("Bearish BOS history", stats.get("bearish_bos", 0))
        c.metric("Qualified historical setups", stats.get("total_qualified_setups", 0))
        st.write("**Active historical setups:**", stats.get("active_setups", 0))
        st.write("**Invalidated historical setups:**", stats.get("invalidated_setups", 0))
        st.write("**Active BSL levels:**", stats.get("active_bsl_levels", 0))
        st.write("**Active SSL levels:**", stats.get("active_ssl_levels", 0))
        setups = result.get("setups", pd.DataFrame())
        if isinstance(setups, pd.DataFrame) and not setups.empty:
            st.write("**Qualified OB + FVG overlap history**")
            st.dataframe(setups.tail(30), use_container_width=True, hide_index=True)
        else:
            st.info("No qualified historical OB/FVG overlap was detected in this data window.")
        liquidity = result.get("liquidity", pd.DataFrame())
        if isinstance(liquidity, pd.DataFrame) and not liquidity.empty:
            st.write("**BSL / SSL historical levels**")
            st.dataframe(liquidity.tail(30), use_container_width=True, hide_index=True)

# Engine details
st.divider()
with st.expander("Data and engine information"):
    details = [
        "Market data provider: Yahoo Finance via yfinance.",
        "4H bars are resampled from 1H data.",
        "Newest candle is conservatively omitted from each downloaded interval to reduce partial-candle use.",
        "Each timeframe is analysed independently.",
        "Historical structure analysis is performed by strategy_engine.py.",
        "The chart displays the shared OB/FVG overlap area; standalone FVGs are not drawn.",
        "S/R invalidation is based on completed-candle close beyond a zone's outer boundary on its source timeframe.",
        "S/R layers: 4H major, 1H major, 15M minor.",
        "Analysis uses up to 1,200 candles per timeframe. The chart displays up to 1,200 candles and initially focuses on the latest 60.",
        "No fabricated candles are created when data is unavailable.",
        "This app is for historical education and virtual review only; it does not place orders or connect to a broker.",
    ]
    for detail in details:
        st.write(detail)

st.caption("DESTINY AI | Historical analysis and virtual review")
