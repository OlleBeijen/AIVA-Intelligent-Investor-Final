
import streamlit as st
import pandas as pd
from aiva_core.data_sources import fetch_prices
from aiva_core.ml_forecast import forecast_ml

st.set_page_config(page_title="ML-Voorspelling – AIVA", page_icon="🤖", layout="wide")
st.title("🤖 ML-Voorspelling (Gradient Boosting)")

tickers = st.text_input("Tickers", value="AAPL,MSFT,ASML.AS")
horizon = st.slider("Horizon (dagen)", 3, 20, 5)
lookback = st.slider("Lookback (dagen aan data)", 180, 1500, 365)

if st.button("Train & Voorspel"):
    ts = [t.strip() for t in tickers.split(",") if t.strip()]
    with st.spinner("Data ophalen en model trainen..."):
        prices = fetch_prices(ts, lookback_days=lookback)
        res = forecast_ml(prices, horizon=horizon)
    if not res:
        st.info("Geen voorspellingen (onvoldoende data?).")
    else:
        st.json(res)
