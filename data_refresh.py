
import os
import time
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf
import requests

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

UPLOAD_URL = os.getenv("SUPABASE_UPLOAD_URL")
UPLOAD_TOKEN = os.getenv("DATA_UPLOAD_TOKEN")


def refresh_market_data():
    if not UPLOAD_URL or not UPLOAD_TOKEN:
        logging.error("Supabase upload configuration is missing")
        return

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

            filename = f"{name.lower().replace(' ', '_')}.csv"
            filepath = DATA_DIR / filename
            df.to_csv(filepath)

            response = requests.post(
                UPLOAD_URL,
                headers={
                    "x-upload-token": UPLOAD_TOKEN,
                    "Content-Type": "application/json",
                },
                json={
                    "filename": filename,
                    "content": filepath.read_text(encoding="utf-8"),
                },
                timeout=60,
            )

            response.raise_for_status()

            logging.info(
                "%s: %s rows uploaded successfully",
                symbol,
                len(df),
            )

        except Exception:
            logging.exception("Refresh failed for %s", symbol)


if __name__ == "__main__":
    logging.info("Destiny AI data refresh service started")

    while True:
        refresh_market_data()
        logging.info("Next refresh in 15 minutes")
        time.sleep(REFRESH_SECONDS)
