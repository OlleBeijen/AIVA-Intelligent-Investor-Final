
import streamlit as st
import pandas as pd
from aiva_core.data_sources import fetch_prices
from aiva_core.signals import indicators, generate_signals

st.set_page_config(page_title="Signalen – AIVA", page_icon="🚦", layout="wide")
st.title("🚦 Signalen & Regimes")

tickers = st.text_input("Tickers", value="AAPL,MSFT,ASML.AS")
lookback = st.slider("Lookback (dagen)", 120, 1000, 365)
rsi_buy = st.number_input("RSI koop ≤", 10, 60, 30)
rsi_sell = st.number_input("RSI verkoop ≥", 40, 90, 70)
sma_s = st.number_input("SMA kort", 5, 200, 20)
sma_l = st.number_input("SMA lang", 10, 400, 50)

if st.button("Genereer"):
    ts = [t.strip() for t in tickers.split(",") if t.strip()]
    prices = fetch_prices(ts, lookback_days=lookback)
    all_rows = []
    for t, df in prices.items():
        ind = indicators(df["Close"], sma_s=int(sma_s), sma_l=int(sma_l), rsi_n=14, bb_n=20, macd_fast=12, macd_slow=26, macd_signal=9)
        sigs = generate_signals(ind, rsi_buy=int(rsi_buy), rsi_sell=int(rsi_sell))
        sig_last = sigs.iloc[-1] if len(sigs) else None
        all_rows.append({"ticker": t, "signal": str(sig_last) if sig_last is not None else "NA"})
    st.dataframe(pd.DataFrame(all_rows), use_container_width=True)
