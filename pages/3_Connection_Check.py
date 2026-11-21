import streamlit as st
import urllib.request

st.set_page_config(page_title="Connection Check", layout="centered")

st.title("DESTINY AI")
st.subheader("Supabase Connection Check")

try:
    url = st.secrets["supabase"]["url"].rstrip("/")
    key = st.secrets["supabase"]["key"]

    request = urllib.request.Request(
        f"{url}/rest/v1/",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}"
        }
    )

    with urllib.request.urlopen(request, timeout=15) as response:
        if response.status == 200:
            st.success("Supabase connection successful!")
        else:
            st.warning(f"Server response: {response.status}")

except Exception as error:
    st.error("Connection test failed.")
    st.caption(str(error))
