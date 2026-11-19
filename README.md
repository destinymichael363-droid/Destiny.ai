# Destiny AI — Android-friendly prototype (v2)

## What it is
A Python Streamlit web app that can be opened in an Android browser after it is deployed to a Python hosting service. It is not an APK.

## Included
- Separate CSV uploads for 1H, 15M, 5M and 1M
- MT5-style candle column normalization
- ATR(14) calculation
- Confirmed 5-bar swing pivots (right-side confirmation)
- Basic HH/HL and LH/LL structure classification
- Candidate close-break events against the latest confirmed swing
- ATR-height pivot-zone candidates
- Three-candle FVG candidates
- Virtual trade journal with price-order validation, planned R:R calculation and CSV export

## Not yet included
- Exact port of the full custom Pine Script indicators
- Your S/R zone touch/rejection count and confirmed flip-zone state machine
- BOS-qualified OB + FVG pairing, displacement, volume qualification and paired invalidation
- Automatic alignment of candles across timeframes
- Automated historical backtesting and performance statistics
- Persistent journal storage after a browser session ends
- Direct connection to MT5 Android or broker order execution

## Run
Install requirements and run:
`streamlit run app.py`

## CSV
Required: `open`, `high`, `low`, `close`.
Recommended: `time`, `tick_volume`.
Common MT5 export headings are recognized.

## Important
The rules here are intentionally labelled as basic candidates. Do not treat them as financial advice or verified signals. Forex tick volume is not centralized traded volume. No orders are placed.
