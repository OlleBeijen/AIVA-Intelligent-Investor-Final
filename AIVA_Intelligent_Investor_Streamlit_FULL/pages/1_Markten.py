
import streamlit as st
import pandas as pd
from utils.data import fetch_bulk_quotes

st.set_page_config(page_title="Markten – AIVA", page_icon="📊", layout="wide")

st.title("📊 Markten – Live overzicht")
st.caption("Tip: zet je favoriete tickers in de sidebar op de startpagina.")

major = ["^GSPC","^IXIC","^DJI","^RUT","^VIX","CL=F","GC=F","SI=F","EURUSD=X","BTC-USD","ETH-USD"]
user = st.text_input("Voeg tickers toe (kommagescheiden)", value="ASML.AS,PHIA.AS,AD.AS")
tickers = major + [t.strip() for t in user.split(",") if t.strip()]

dfq = fetch_bulk_quotes(tickers)
st.dataframe(dfq, use_container_width=True)
