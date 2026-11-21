
import streamlit as st
from supabase import create_client


def get_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def sign_up(email, password):
    supabase = get_supabase()
    return supabase.auth.sign_up({
        "email": email,
        "password": password
    })


def sign_in(email, password):
    supabase = get_supabase()
    response = supabase.auth.sign_in_with_password({
        "email": email,
        "password": password
    })

    if response.session:
        st.session_state["supabase_access_token"] = (
            response.session.access_token
        )
        st.session_state["supabase_refresh_token"] = (
            response.session.refresh_token
        )
        st.session_state["supabase_user"] = response.user.id

    return response


def sign_out():
    st.session_state.pop("supabase_access_token", None)
    st.session_state.pop("supabase_refresh_token", None)
    st.session_state.pop("supabase_user", None)


def is_authenticated():
    return bool(st.session_state.get("supabase_user"))

