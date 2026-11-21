import streamlit as st

def require_login():
    if not st.session_state.get("supabase_user"):
        st.warning("Please log in to access Destiny AI.")

        st.page_link("pages/1_Login.py", label="Go to Login")

        st.stop()
