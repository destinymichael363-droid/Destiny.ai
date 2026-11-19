import streamlit as st
import pandas as pd
from destiny.engine import normalize_columns, analyze_market

st.set_page_config(page_title="Destiny AI", page_icon="📊", layout="wide")
st.title("📊 Destiny AI")
st.caption("Multi-timeframe market-structure workspace • educational analysis • virtual journal")

st.warning("Analysis only. No broker orders are sent. Findings are candidates for chart review, not instructions or guaranteed signals.")

with st.sidebar:
    st.header("Import MT5 candle history")
    st.caption("Upload one CSV per timeframe. Files should contain time, open, high, low, close and optionally tick_volume.")
    uploads = {}
    for tf in ["1H", "15M", "5M", "1M"]:
        uploads[tf] = st.file_uploader(f"{tf} candles", type=["csv"], key=f"file_{tf}")

datasets = {}
errors = {}
for tf, uploaded in uploads.items():
    if uploaded is not None:
        try:
            datasets[tf] = normalize_columns(pd.read_csv(uploaded))
        except Exception as e:
            errors[tf] = str(e)

if errors:
    for tf, err in errors.items():
        st.error(f"{tf}: {err}")

if not datasets:
    st.subheader("Start by importing candle data")
    st.markdown("""
    **On your MT5 Android app:** use an available export route to obtain historical candle CSV files. The Android MT5 app itself does not provide the official desktop Python bridge.

    Upload each timeframe separately in the sidebar. You can begin with just 1H, then add 15M, 5M and 1M.

    **CSV columns required:** `open`, `high`, `low`, `close`. A time column and tick volume are recommended.
    """)
    st.stop()

st.subheader("Multi-timeframe overview")
cols = st.columns(4)
for i, tf in enumerate(["1H", "15M", "5M", "1M"]):
    with cols[i]:
        if tf in datasets:
            r = analyze_market(datasets[tf])
            st.metric(tf, r["structure"], f'{len(datasets[tf])} candles')
        else:
            st.metric(tf, "Not loaded", "Upload CSV")

tabs = st.tabs([tf for tf in ["1H", "15M", "5M", "1M"] if tf in datasets])
for tab, tf in zip(tabs, [tf for tf in ["1H", "15M", "5M", "1M"] if tf in datasets]):
    with tab:
        df = datasets[tf]
        r = analyze_market(df)
        x, y, z = st.columns(3)
        x.metric("Structure", r["structure"])
        y.metric("Review status", r["status"])
        z.metric("Latest close", f'{df.close.iloc[-1]:.6g}')
        st.write(r["summary"])
        st.markdown("**Structure explanation**")
        st.write(r["structure_detail"])
        left, right = st.columns(2)
        with left:
            st.markdown("**ATR-height pivot-zone candidates**")
            st.dataframe(pd.DataFrame(r["zones"]), use_container_width=True)
            st.markdown("**Break-of-structure candidates**")
            st.dataframe(pd.DataFrame(r["bos"]), use_container_width=True)
        with right:
            st.markdown("**Fair-value-gap candidates**")
            st.dataframe(pd.DataFrame(r["fvg"]), use_container_width=True)
            st.markdown("**Recent candles**")
            st.dataframe(df.tail(12).iloc[::-1], use_container_width=True)

st.divider()
st.subheader("Virtual trade journal")
st.caption("For practice and strategy evaluation only. Entries and outcomes are hypothetical; this does not connect to MT5 execution.")
if "journal" not in st.session_state:
    st.session_state.journal = []

with st.form("paper_trade_form"):
    c1, c2, c3 = st.columns(3)
    symbol = c1.text_input("Symbol", placeholder="e.g. EURUSD")
    side = c2.selectbox("Direction", ["Buy", "Sell"])
    timeframe = c3.selectbox("Review timeframe", ["1H", "15M", "5M", "1M"])
    c4, c5, c6 = st.columns(3)
    entry = c4.number_input("Hypothetical entry", min_value=0.0, format="%.6f")
    stop = c5.number_input("Hypothetical invalidation level", min_value=0.0, format="%.6f")
    target = c6.number_input("Hypothetical target", min_value=0.0, format="%.6f")
    notes = st.text_input("Reason / setup notes")
    submitted = st.form_submit_button("Add to virtual journal")
    if submitted:
        if not symbol.strip() or min(entry, stop, target) <= 0:
            st.error("Enter a symbol and positive price values.")
        elif (side == "Buy" and not stop < entry < target) or (side == "Sell" and not target < entry < stop):
            st.error("Check the price order: for Buy, stop < entry < target; for Sell, target < entry < stop.")
        else:
            risk = abs(entry-stop)
            reward = abs(target-entry)
            st.session_state.journal.append({
                "Symbol": symbol.upper().strip(), "Side": side, "TF": timeframe,
                "Entry": entry, "Invalidation": stop, "Target": target,
                "Planned R:R": round(reward/risk, 2), "Notes": notes, "Status": "Open (virtual)"
            })
            st.success("Added to the virtual journal.")

if st.session_state.journal:
    journal_df = pd.DataFrame(st.session_state.journal)
    st.dataframe(journal_df, use_container_width=True)
    st.download_button("Download journal CSV", journal_df.to_csv(index=False), "destiny_virtual_journal.csv", "text/csv")
    if st.button("Clear virtual journal"):
        st.session_state.journal = []
        st.rerun()
else:
    st.info("No virtual records yet. Add a hypothetical setup above to begin a practice journal.")

st.divider()
st.caption("Current build: CSV import, basic confirmed pivots, simple structure classification, candidate BOS/FVG zones and session-only virtual journal. Exact Pine Script parity, automated historical backtesting and persistent cloud storage are still future development tasks.")
