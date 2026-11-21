
import streamlit as st
import pandas as pd
import urllib.request
from datetime import datetime, timezone


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Destiny AI | Paper Account",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    padding-left: 1rem;
    padding-right: 1rem;
    max-width: 100%;
}
h1 {font-size: 1.55rem !important;}
h2, h3 {font-size: 1.15rem !important;}
</style>
""", unsafe_allow_html=True)


# ==========================================
# TIME FUNCTION
# ==========================================

def current_time():
    return datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M"
    )


# ==========================================
# ANDROID NOTIFICATION
# ==========================================

def send_app_notification(title, message):
    try:
        topic = st.secrets.get("NTFY_TOPIC", "").strip()

        if not topic:
            return False, "NTFY_TOPIC is missing from Secrets."

        request = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": title},
            method="POST",
        )

        with urllib.request.urlopen(
            request, timeout=15
        ) as response:
            if response.status in (200, 201):
                return True, "Notification sent."

            return False, f"Server returned status {response.status}."

    except Exception as error:
        return False, str(error)


# ==========================================
# PAGE TITLE
# ==========================================

st.title("DESTINY AI - PAPER ACCOUNT")

st.caption(
    "A virtual account for learning and historical testing."
)

st.warning(
    "SIMULATION ONLY - No bank, broker, wallet or payment "
    "service is connected. All balances are virtual."
)


# ==========================================
# INITIAL ACCOUNT
# ==========================================

if "paper_balance" not in st.session_state:
    st.session_state.paper_balance = 10000.00

if "paper_transactions" not in st.session_state:
    st.session_state.paper_transactions = [{
        "Time (UTC)": current_time(),
        "Type": "Starting virtual balance",
        "Amount": 10000.00,
        "Virtual balance": 10000.00,
    }]


# ==========================================
# ACCOUNT OVERVIEW
# ==========================================

st.subheader("Virtual account overview")

col1, col2 = st.columns(2)

col1.metric(
    "Virtual balance",
    f"${st.session_state.paper_balance:,.2f}"
)

col2.metric(
    "Account type",
    "Paper only"
)

st.divider()


# ==========================================
# VIRTUAL ADJUSTMENTS
# ==========================================

st.subheader("Simulation controls")

st.write(
    "Adjust your virtual practice balance. "
    "These are not real financial transactions."
)

with st.form("paper_adjustment"):

    action = st.radio(
        "Choose a virtual adjustment",
        [
            "Add virtual funds",
            "Remove virtual funds",
        ],
    )

    amount = st.number_input(
        "Amount (virtual USD)",
        min_value=1.0,
        max_value=1000000.0,
        value=100.0,
        step=10.0,
    )

    submitted = st.form_submit_button(
        "Apply virtual adjustment",
        use_container_width=True,
    )


# ==========================================
# PROCESS ADJUSTMENT
# ==========================================

if submitted:

    amount = float(amount)

    if action == "Add virtual funds":

        st.session_state.paper_balance += amount
        transaction_type = "Virtual funds added"
        signed_amount = amount

    elif amount <= st.session_state.paper_balance:

        st.session_state.paper_balance -= amount
        transaction_type = "Virtual funds removed"
        signed_amount = -amount

    else:

        st.error(
            "The adjustment cannot exceed your current "
            "virtual balance."
        )
        st.stop()

    st.session_state.paper_transactions.insert(0, {
        "Time (UTC)": current_time(),
        "Type": transaction_type,
        "Amount": signed_amount,
        "Virtual balance": st.session_state.paper_balance,
    })

    st.success("Virtual account updated.")

    notification_ok, notification_message = (
        send_app_notification(
            "Destiny AI - Paper Account",
            (
                f"{transaction_type}\n"
                f"Amount: ${amount:,.2f}\n"
                f"New virtual balance: "
                f"${st.session_state.paper_balance:,.2f}"
            ),
        )
    )

    if notification_ok:
        st.success("Android notification sent.")
    else:
        st.warning(
            "Account updated, but notification failed."
        )
        st.caption(notification_message)


st.divider()


# ==========================================
# TRANSACTION HISTORY
# ==========================================

st.subheader("Virtual transaction history")

history = pd.DataFrame(
    st.session_state.paper_transactions
)

st.dataframe(
    history,
    use_container_width=True,
    hide_index=True,
)


# ==========================================
# CSV DOWNLOAD
# ==========================================

csv_data = history.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "Download virtual history (CSV)",
    data=csv_data,
    file_name="destiny_ai_paper_account_history.csv",
    mime="text/csv",
    use_container_width=True,
)


# ==========================================
# RESET ACCOUNT
# ==========================================

if st.button(
    "Reset virtual account to $10,000",
    use_container_width=True,
):

    st.session_state.paper_balance = 10000.00

    st.session_state.paper_transactions = [{
        "Time (UTC)": current_time(),
        "Type": "Virtual account reset",
        "Amount": 10000.00,
        "Virtual balance": 10000.00,
    }]

    st.rerun()


# ==========================================
# FOOTER
# ==========================================

st.caption(
    "Virtual account values are stored in the current "
    "Streamlit session and may reset when the session restarts. "
    "This is not a live broker account."
)
