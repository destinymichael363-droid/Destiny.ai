import streamlit as st
from supabase import create_client

st.set_page_config(page_title="Destiny AI Connection", layout="centered")

st.title("DESTINY AI")
st.subheader("Database Connection")

try:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]

    supabase = create_client(url, key)

    st.success("Supabase client initialized successfully!")
    st.info("Connection configuration is loaded correctly.")

except Exception as error:
    st.error("Connection initialization failed.")
    st.caption(str(error))
