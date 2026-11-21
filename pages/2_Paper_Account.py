import secrets
from auth_guard import require_login
from auth import get_authenticated_supabase

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

require_login()

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
# DATABASE CONNECTION
# ==========================================

try:
    supabase = get_authenticated_supabase()

    if supabase is None:
        st.error("Your login session is unavailable. Please log in again.")
        st.stop()

    user_id = st.session_state.get("supabase_user")

    if not user_id:
        st.error("User account could not be identified.")
        st.stop()

except Exception as error:
    st.error("Could not connect to your authenticated database.")
    st.caption(str(error))
    st.stop()


# ==========================================
# LOAD OR CREATE USER ACCOUNT
# ==========================================

def load_paper_account():

    result = (
        supabase
        .table("paper_account")
        .select("balance, transactions, notification_topic")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )

    if result.data:
    account = result.data[0]

    if not account.get("notification_topic"):
        topic = secrets.token_urlsafe(32)

        supabase.table("paper_account").update({
            "notification_topic": topic
        }).eq("user_id", user_id).execute()

        account["notification_topic"] = topic

    return account
        return result.data[0]

    initial_history = [{
        "Time (UTC)": current_time(),
        "Type": "Starting virtual balance",
        "Amount": 10000.00,
        "Virtual balance": 10000.00,
    }]

    new_account = {
        "user_id": user_id,
        "balance": 10000.00,
        "transactions": initial_history,
    }

    created = (
        supabase
        .table("paper_account")
        .insert(new_account)
        .execute()
    )

    if not created.data:
        raise Exception("Could not create your virtual account.")

    return created.data[0]


def save_paper_account(balance, transactions):

    result = (
        supabase
        .table("paper_account")
        .update({
            "balance": float(balance),
            "transactions": transactions,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        .eq("user_id", user_id)
        .execute()
    )

    if not result.data:
        raise Exception(
            "Database did not confirm the account update."
        )


try:
    account = load_paper_account()

    paper_balance = float(account["balance"])
    paper_transactions = account["transactions"] or []

except Exception as error:
    st.error("Could not load your virtual account.")
    st.caption(str(error))
    st.stop()


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
# ACCOUNT OVERVIEW
# ==========================================

st.subheader("Virtual account overview")

col1, col2 = st.columns(2)

col1.metric(
    "Virtual balance",
    f"${paper_balance:,.2f}"
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

        new_balance = paper_balance + amount
        transaction_type = "Virtual funds added"
        signed_amount = amount

    elif amount <= paper_balance:

        new_balance = paper_balance - amount
        transaction_type = "Virtual funds removed"
        signed_amount = -amount

    else:

        st.error(
            "The adjustment cannot exceed your current "
            "virtual balance."
        )
        st.stop()

    new_history = [{
        "Time (UTC)": current_time(),
        "Type": transaction_type,
        "Amount": signed_amount,
        "Virtual balance": new_balance,
    }] + paper_transactions

    try:

        save_paper_account(new_balance, new_history)

        st.success("Virtual account saved successfully.")

        notification_ok, notification_message = (
            send_app_notification(
                "Destiny AI - Paper Account",
                (
                    f"{transaction_type}\n"
                    f"Amount: ${amount:,.2f}\n"
                    f"New virtual balance: ${new_balance:,.2f}"
                ),
            )
        )

        if notification_ok:
            st.success("Android notification sent.")
        else:
            st.warning(
                "Account saved, but notification failed."
            )
            st.caption(notification_message)

        st.rerun()

    except Exception as error:

        st.error("Could not save your virtual adjustment.")
        st.caption(str(error))


st.divider()


# ==========================================
# TRANSACTION HISTORY
# ==========================================

st.subheader("Virtual transaction history")

history = pd.DataFrame(paper_transactions)

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

    reset_history = [{
        "Time (UTC)": current_time(),
        "Type": "Virtual account reset",
        "Amount": 10000.00,
        "Virtual balance": 10000.00,
    }]

    try:

        save_paper_account(10000.00, reset_history)

        st.success("Virtual account reset successfully.")
        st.rerun()

    except Exception as error:

        st.error("Could not reset your virtual account.")
        st.caption(str(error))


# ==========================================
# FOOTER
# ==========================================

st.caption(
    "Virtual account balance and history are stored "
    "in your Supabase account. This is not a live broker account."
)
