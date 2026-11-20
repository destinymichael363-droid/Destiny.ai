import streamlit as st
import pandas as pd
from datetime import datetime, timezone

st.set_page_config(
    page_title="Destiny AI | Paper Account",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container {padding-top:1rem;padding-bottom:2rem;padding-left:1rem;padding-right:1rem;max-width:100%;}
h1 {font-size:1.55rem !important;}
h2,h3 {font-size:1.15rem !important;}
</style>
""", unsafe_allow_html=True)

st.title("DESTINY AI — PAPER ACCOUNT")
st.caption("A virtual account for learning and historical strategy testing.")

st.warning(
    "SIMULATION ONLY — This page does not connect to a bank, broker, wallet, "
    "or payment service. No real money can be deposited, withdrawn, or traded."
)

if "paper_balance" not in st.session_state:
    st.session_state.paper_balance = 10000.00
if "paper_transactions" not in st.session_state:
    st.session_state.paper_transactions = [{
        "Time (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "Type": "Starting virtual balance",
        "Amount": 10000.00,
        "Virtual balance": 10000.00,
    }]

st.subheader("Virtual account overview")
a, b = st.columns(2)
a.metric("Virtual balance", f"${st.session_state.paper_balance:,.2f}")
b.metric("Account type", "Paper only")

st.divider()
st.subheader("Simulation controls")
st.write("You can adjust the practice balance to test how the app displays account records. These changes are virtual and are not financial transactions.")

with st.form("paper_adjustment"):
    action = st.radio("Choose a virtual adjustment", ["Add virtual funds", "Remove virtual funds"], horizontal=False)
    amount = st.number_input("Amount (virtual USD)", min_value=1.0, max_value=1000000.0, value=100.0, step=10.0)
    submitted = st.form_submit_button("Apply virtual adjustment", use_container_width=True)

if submitted:
    if action == "Add virtual funds":
        st.session_state.paper_balance += float(amount)
        tx_type = "Virtual funds added"
        signed_amount = float(amount)
    elif float(amount) <= st.session_state.paper_balance:
        st.session_state.paper_balance -= float(amount)
        tx_type = "Virtual funds removed"
        signed_amount = -float(amount)
    else:
        st.error("The virtual adjustment cannot exceed the current practice balance.")
        st.stop()

    st.session_state.paper_transactions.insert(0, {
        "Time (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "Type": tx_type,
        "Amount": signed_amount,
        "Virtual balance": st.session_state.paper_balance,
    })
    st.success("Virtual account updated.")

st.divider()
st.subheader("Virtual transaction history")
history = pd.DataFrame(st.session_state.paper_transactions)
st.dataframe(history, use_container_width=True, hide_index=True)

csv_data = history.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download virtual history (CSV)",
    data=csv_data,
    file_name="destiny_ai_paper_account_history.csv",
    mime="text/csv",
    use_container_width=True,
)

if st.button("Reset virtual account to $10,000", use_container_width=True):
    st.session_state.paper_balance = 10000.00
    st.session_state.paper_transactions = [{
        "Time (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "Type": "Virtual account reset",
        "Amount": 10000.00,
        "Virtual balance": 10000.00,
    }]
    st.rerun()

st.caption(
    "Virtual account values are stored in this Streamlit session and may reset "
    "when the app session restarts. This is not a live broker account."
)

