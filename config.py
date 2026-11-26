"""
config.py — Single source of truth for DESTINY AI.
"""

from __future__ import annotations

APP_NAME = "DESTINY AI"
BASE_TF_MIN = 15

WORKER_WATCHLIST: list[tuple[str, str]] = [
    ("Gold (XAU)", "GC=F"),
    ("Crude Oil (WTI)", "CL=F"),
    ("EUR/USD", "EURUSD=X"),
    ("GBP/USD", "GBPUSD=X"),
    ("Bitcoin (BTC)", "BTC-USD"),
]

STRATEGY_TIMEFRAMES: list[str] = [
    "4H",
    "1H",
    "15M",
    "5M",
    "1M",
]

DEFAULT_BALANCE = 10_000.0
DEFAULT_RISK_PCT = 2.0
DEFAULT_RR_TARGET = 5.0

MAX_AUTO_OPEN_POSITIONS = 3
MAX_TOTAL_RISK_PCT = 6.5
MAX_RISK_PCT_PER_TRADE = 5.0
MAX_LEVERAGE = 20.0
MARGIN_RATE = 0.05

CORRELATED_GROUPS: list[set[str]] = [
    {"EURUSD=X", "GBPUSD=X"},
]

INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
}

MAX_SIGNALS = 25
MAX_CLOSED_LOG = 500

OHLCV: list[str] = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]
