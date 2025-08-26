
import streamlit as st
import pandas as pd
from aiva_core.value_screen import compute_scores

st.set_page_config(page_title="Waardering – AIVA", page_icon="💎", layout="wide")
st.title("💎 Waardering & Value-score")
st.caption("Combineert PE, PB, EV/EBITDA, FCF en eenvoudige DCF (indien data).")

tickers = st.text_input("Tickers", value="AAPL,MSFT,GOOGL,AMZN,TSLA,ASML.AS,AD.AS")
if st.button("Bereken"):
    ts = [t.strip() for t in tickers.split(",") if t.strip()]
    df = compute_scores(ts)
    if isinstance(df, pd.DataFrame) and not df.empty:
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "waardering.csv", "text/csv")
    else:
        st.info("Geen resultaten.")
