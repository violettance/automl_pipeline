# test_app.py
import streamlit as st
import sys
import os
import platform

st.title("Streamlit Environment Check")

st.write(f"**Python Executable:** `{sys.executable}`")
st.write(f"**Platform:** `{platform.platform()}`")
st.write(f"**Python Version:** `{sys.version}`")
st.write(f"**Streamlit Version:** `{st.__version__}`")

st.subheader("sys.path:")
st.json(sys.path)

st.subheader("Environment Variables:")
st.json({k: v for k, v in os.environ.items() if 'PYTHON' in k or 'CONDA' in k or 'PATH' in k})

st.subheader("Import Test:")
try:
    import ydata_profiling
    st.success(f"Successfully imported `ydata_profiling` version `{ydata_profiling.__version__}`")
    st.write(f"Location: `{ydata_profiling.__file__}`")
except ImportError as e:
    st.error(f"Failed to import `ydata_profiling`: {e}")
except Exception as e:
    st.error(f"An unexpected error occurred during import: {e}")

st.write("Check complete.")