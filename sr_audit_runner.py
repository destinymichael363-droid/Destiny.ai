# =========================================================
# DESTINY AI - S/R AUDIT RUNNER
# Separate Streamlit page; does not modify the main app.
# =========================================================
import streamlit as st
import io
import contextlib
import importlib

st.set_page_config(page_title="Destiny AI — S/R Audit", layout="wide")
st.title("Destiny AI — S/R Engine Audit")
st.caption("Diagnostic only • synthetic candles • no market data or trade signals")
st.write("This page runs the separate S/R audit script against the repository's sr_engine.py.")

if st.button("Run S/R audit", type="primary"):
    output = io.StringIO()
    try:
        audit = importlib.import_module("sr_audit")
        with contextlib.redirect_stdout(output):
            audit.main()
        st.code(output.getvalue(), language="text")
        if "[FAIL]" in output.getvalue():
            st.error("Audit encountered a failure. Review the output above.")
        elif "[CHECK]" in output.getvalue():
            st.warning("Audit completed with diagnostic CHECK items. These are flags to review, not automatically defects.")
        else:
            st.success("Audit completed. Review the results above.")
    except Exception as exc:
        st.error(f"Audit runner error: {exc}")
        st.code(output.getvalue(), language="text")
else:
    st.info("Tap **Run S/R audit** to begin the diagnostic.")

