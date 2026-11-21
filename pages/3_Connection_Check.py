
import streamlit as st
from supabase import create_client

st.set_page_config(page_title="Destiny AI Connection", layout="centered")

st.title("DESTINY AI")
st.subheader("Database Verification")

try:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]

    supabase = create_client(url, key)

    st.success("Supabase client initialized.")

    # Test access to the paper_account table
    result = supabase.table("paper_account").select("id").limit(1).execute()

    st.success("Database query successful!")
    st.write("Paper account table is responding.")
    
except Exception as error:
    st.error("Database verification failed.")
    st.write(str(error))
