
import os
import time
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

WATCHLIST = {
    "Gold": "GC=F",
    "Crude Oil": "CL=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "Bitcoin": "BTC-USD",
}

REFRESH_SECONDS = 900
DATA_DIR = Path("data/market_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def refresh_market_data():
    for name, symbol in WATCHLIST.items():
        try:
            df = yf.download(
                symbol,
                period="14d",
                interval="15m",
                progress=False,
                auto_adjust=False,
            )

            if df.empty:
                logging.warning("%s: no data returned", symbol)
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.dropna()
            filename = DATA_DIR / f"{name.lower().replace(' ', '_')}.csv"
            df.to_csv(filename)

            logging.info(
                "%s: saved %s rows to %s",
                symbol,
                len(df),
                filename,
            )

        except Exception:
            logging.exception("Refresh failed for %s", symbol)


if __name__ == "__main__":
    logging.info("Destiny AI data refresh service started")

    while True:
        refresh_market_data()
        logging.info("Next refresh in 15 minutes")
        time.sleep(REFRESH_SECONDS)
