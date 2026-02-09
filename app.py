
import streamlit as st
import pathlib
import sys

# Add project root to path
sys.path.append(str(pathlib.Path(__file__).parent))

st.set_page_config(page_title="AI Stock Predictor", page_icon="📈")

st.title("AI Stock Predictor 🚀")
st.info("This app has been updated to support ANY stock ticker!")

st.markdown("""
### ⚠️ Notice
You are running `app.py`, which was the old entry point. 
Please usage **`streamlit_app.py`** instead for the latest features.

**To run the new app:**
```bash
streamlit run streamlit_app.py
```
""")

if st.button("Redirect to New App (Simulation)"):
    st.markdown("Please stop this process and run `streamlit run streamlit_app.py` in your terminal.")
