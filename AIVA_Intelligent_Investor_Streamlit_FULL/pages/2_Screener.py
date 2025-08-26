
import streamlit as st
import pandas as pd
from utils.data import fetch_history
from utils.indicators import sma, rsi

st.set_page_config(page_title="Screener – AIVA", page_icon="🧭", layout="wide")

st.title("🧭 Simpele Screener")
st.caption("Scan een lijst met tickers op momentum / trendregels.")

tickers = st.text_area("Tickers (kommagescheiden)", value="AAPL,MSFT,NVDA,AMZN,GOOGL,TSLA,ASML.AS,PHIA.AS,AD.AS").split(",")
tickers = [t.strip() for t in tickers if t.strip()]

col1, col2, col3 = st.columns(3)
with col1:
    period = st.selectbox("Periode", ["6mo","1y","2y"], index=1)
with col2:
    fast = st.number_input("SMA kort", min_value=5, max_value=200, value=20, step=1)
with col3:
    slow = st.number_input("SMA lang", min_value=10, max_value=400, value=50, step=1)

rows = []
progress = st.progress(0.0)
for i, t in enumerate(tickers):
    df = fetch_history(t, period=period, interval="1d")
    if df.empty:
        rows.append({"ticker": t, "close": None, "above_sma": None, "sma_fast": None, "sma_slow": None, "rsi14": None})
    else:
        df["SMA_F"] = sma(df["Close"], fast)
        df["SMA_S"] = sma(df["Close"], slow)
        df["RSI14"] = rsi(df["Close"], 14)
        last = df.iloc[-1]
        rows.append({
            "ticker": t,
            "close": float(last["Close"]),
            "above_sma": bool(last["SMA_F"] > last["SMA_S"]) if pd.notna(last["SMA_F"]) and pd.notna(last["SMA_S"]) else None,
            "sma_fast": float(last["SMA_F"]) if pd.notna(last["SMA_F"]) else None,
            "sma_slow": float(last["SMA_S"]) if pd.notna(last["SMA_S"]) else None,
            "rsi14": float(last["RSI14"]) if pd.notna(last["RSI14"]) else None,
        })
    progress.progress((i+1)/len(tickers))

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True)
st.download_button("Exporteer resultaten (CSV)", df.to_csv(index=False).encode("utf-8"), file_name="screener_resultaten.csv", mime="text/csv")
