
import os
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

DATA_DIR = Path("data/market_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_URL = os.getenv("SUPABASE_UPLOAD_URL")
UPLOAD_TOKEN = os.getenv("DATA_UPLOAD_TOKEN")


def refresh_market_data():
    if not UPLOAD_URL or not UPLOAD_TOKEN:
        raise RuntimeError("Supabase upload configuration is missing")

    failures = []
    successful_uploads = 0

    for name, symbol in WATCHLIST.items():
        try:
            logging.info("Downloading %s", symbol)

            df = yf.download(
                symbol,
                period="14d",
                interval="15m",
                progress=False,
                auto_adjust=False,
            )

            if df.empty:
                raise RuntimeError("No market data returned")

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

            if not response.ok:
                raise RuntimeError(
                    f"Upload HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )

            successful_uploads += 1
            logging.info(
                "%s: %s rows uploaded successfully",
                symbol,
                len(df),
            )

        except Exception as error:
            logging.exception("Failed: %s", symbol)
            failures.append(f"{symbol}: {error}")

    logging.info("Successful uploads: %s", successful_uploads)

    if failures:
        raise RuntimeError(
            "Some uploads failed: " + "; ".join(failures)
        )

    logging.info("All five market-data uploads completed")


if __name__ == "__main__":
    logging.info("Destiny AI data refresh started")
    refresh_market_data()
