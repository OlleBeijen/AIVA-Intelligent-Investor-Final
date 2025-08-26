
import streamlit as st
import pandas as pd
from aiva_core.portfolio_import import parse_positions_csv

st.set_page_config(page_title="Import – AIVA", page_icon="📥", layout="wide")
st.title("📥 Portefeuille importeren (CSV)")
st.caption("Minimaal kolommen: 'ticker', 'qty'. Optioneel: 'avg_price', 'currency'.")

uploaded = st.file_uploader("Kies CSV", type=["csv"])
if uploaded:
    try:
        df = parse_positions_csv(uploaded.read())
        st.success("CSV herkend.")
        st.dataframe(df, use_container_width=True)
        st.download_button("Download genormaliseerd CSV", df.to_csv(index=False).encode("utf-8"), "positions_normalized.csv", "text/csv")
    except Exception as e:
        st.error(f"Kon CSV niet parsen: {e}")
else:
    st.info("Upload een CSV om te beginnen.")
