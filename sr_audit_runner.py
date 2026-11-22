
import streamlit as st
import io
import contextlib
import importlib
import traceback
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(
    page_title="Destiny AI — S/R Audits",
    layout="wide"
)

st.title("Destiny AI — S/R Audits")
st.caption("Synthetic diagnostics only • No live-market signals")


def run_audit(module_name):
    output = io.StringIO()

    try:
        importlib.invalidate_caches()

        if module_name in sys.modules:
            audit = importlib.reload(sys.modules[module_name])
        else:
            audit = importlib.import_module(module_name)

        with contextlib.redirect_stdout(output):
            audit.main()

        results = output.getvalue()
        st.code(results, language="text")

        if "[FAIL]" in results:
            st.error("Audit found a failure. Review the output.")
        else:
            st.success("Audit completed. Review the results.")

    except Exception:
        st.error("Audit could not run.")
        st.code(traceback.format_exc())
        st.code(output.getvalue(), language="text")


if st.button("Run Historical Integrity Audit"):
    run_audit("sr_audit")


if st.button("Run Historical Replay Audit", type="primary"):
    run_audit("sr_replay_audit")
