
import streamlit as st
import io
import contextlib
import importlib
import traceback

st.set_page_config(
    page_title="Destiny AI — S/R Audits",
    layout="wide"
)

st.title("Destiny AI — S/R Audits")
st.caption("Synthetic diagnostics only • No live-market signals")

if st.button("Run Historical Integrity Audit", type="secondary"):
    output = io.StringIO()
    try:
        audit = importlib.import_module("sr_audit")
        importlib.reload(audit)

        with contextlib.redirect_stdout(output):
            audit.main()

        st.code(output.getvalue(), language="text")

    except Exception:
        st.error("Historical audit encountered an error.")
        st.code(traceback.format_exc())

if st.button("Run Historical Replay Audit", type="primary"):
    output = io.StringIO()
    try:
        audit = importlib.import_module("sr_replay_audit")
        importlib.reload(audit)

        with contextlib.redirect_stdout(output):
            audit.main()

        results = output.getvalue()
        st.code(results, language="text")

        if "[FAIL]" in results:
            st.error("Replay audit found a failure.")
        else:
            st.success("Replay audit completed. Review its results.")

    except Exception:
        st.error("Replay audit encountered an error.")
        st.code(traceback.format_exc())
