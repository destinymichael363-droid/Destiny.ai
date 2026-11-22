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
