
import streamlit as st
from auth import sign_up, sign_in, sign_out, is_authenticated

st.set_page_config(page_title="Destiny AI Login", layout="centered")

st.title("DESTINY AI")
st.caption("Account Authentication")

if is_authenticated():
    st.success("You are logged in.")

    if st.button("Logout"):
        sign_out()
        st.rerun()

else:
    option = st.radio("Account", ["Login", "Sign Up"])

    email = st.text_input("Email Address")
    password = st.text_input("Password", type="password")

    if option == "Sign Up":
        if st.button("Create Account"):
            try:
                result = sign_up(email, password)

                if result.user:
                    st.success(
                        "Account created successfully. "
                        "Check your email if confirmation is required."
                    )
            except Exception as error:
                st.error("Registration failed.")
                st.caption(str(error))

    if option == "Login":
        if st.button("Login"):
            try:
                result = sign_in(email, password)

                if result.user:
                    st.success("Login successful!")
                    st.rerun()

            except Exception as error:
                st.error("Login failed.")
                st.caption(str(error))

