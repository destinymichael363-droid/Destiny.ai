
from auth_guard import require_login
import streamlit as st
from auth import get_authenticated_supabase

st.set_page_config(
    page_title="Destiny AI Connection",
    layout="centered"
)

require_login()

st.title("DESTINY AI")
st.subheader("Authenticated Database Verification")

try:
    supabase = get_authenticated_supabase()

    if supabase is None:
        st.error("No valid authenticated session found.")
        st.stop()

    st.success("Authenticated Supabase session verified.")

    user_id = st.session_state.get("supabase_user")

    st.success("Logged-in user identified.")

    # Test the database using the user's access token.
    result = (
        supabase
        .table("paper_account")
        .select("user_id")
        .limit(5)
        .execute()
    )

    rows = result.data or []

    st.success("Authenticated database query successful!")

    if rows:
        ownership_ok = all(
            row.get("user_id") == user_id
            for row in rows
        )

        if ownership_ok:
            st.success(
                "Returned records belong to the logged-in user."
            )
        else:
            st.error(
                "Ownership check failed. Review database policies."
            )
    else:
        st.info(
            "No paper account records found yet. "
            "The authenticated query worked, but ownership "
            "cannot be confirmed until an account record exists."
        )

except Exception as error:
    st.error("Authenticated database verification failed.")
    st.caption(str(error))
