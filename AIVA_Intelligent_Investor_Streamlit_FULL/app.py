
import os
from datetime import datetime
import pandas as pd
import streamlit as st
from utils.data import fetch_quote, fetch_history, safe_number, fetch_bulk_quotes
from utils.indicators import sma, rsi, macd
from utils.plotting import price_chart

st.set_page_config(page_title="AIVA – Intelligent Investor", page_icon="📈", layout="wide")

# Sidebar
st.sidebar.title("AIVA – Intelligent Investor")
st.sidebar.caption("Live koersen • Screener • Portefeuille • Backtest • AI Analist")

# Auto refresh (every 60s) to keep quotes fresh
st.sidebar.checkbox("Auto-refresh elke 60s", value=False, key="auto_refresh")
if st.session_state.get("auto_refresh"):
    st.experimental_rerun  # placeholder; Streamlit Cloud may handle via st_autorefresh in pages

st.sidebar.markdown("---")
default_watch = ["AAPL","MSFT","NVDA","ASML.AS","PHIA.AS","^IXIC","^GSPC","BTC-USD","ETH-USD"]
watchlist = st.sidebar.text_input("Watchlist (kommagescheiden)", value=",".join(default_watch))
tickers = [t.strip() for t in watchlist.split(",") if t.strip()]

st.title("🏠 Start")
st.write("Welkom! Vul een ticker in om direct te kijken, of gebruik de tabs hieronder.")

col = st.columns([2,1,1,1,1,1])
with col[0]:
    ticker = st.text_input("Ticker (bv. AAPL, ASML.AS, BTC)", value="AAPL")
with col[1]:
    period = st.selectbox("Periode", ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","max"], index=5)
with col[2]:
    interval = st.selectbox("Interval", ["1m","5m","15m","30m","60m","1d","1wk","1mo"], index=5)
with col[3]:
    sma_fast = st.number_input("SMA kort", value=20, min_value=2, max_value=300, step=1)
with col[4]:
    sma_slow = st.number_input("SMA lang", value=50, min_value=2, max_value=600, step=1)
with col[5]:
    show_rsi = st.checkbox("Toon RSI", value=False)

# Quote header
q = fetch_quote(ticker)
quote_cols = st.columns(5)
quote_cols[0].metric("Ticker", q["ticker"])
quote_cols[1].metric("Prijs", f'{safe_number(q["price"])} {q["currency"]}' if q["price"] else "n.v.t.")
quote_cols[2].metric("Verandering", safe_number(q["change"]))
quote_cols[3].metric("Verandering %", f'{safe_number(q["change_pct"])}%')
quote_cols[4].metric("Markt", q["market_state"])

df = fetch_history(ticker, period=period, interval=interval)
if not df.empty:
    df["SMA_Fast"] = sma(df["Close"], sma_fast)
    df["SMA_Slow"] = sma(df["Close"], sma_slow)

    fig = price_chart(df, title=f"Prijs: {q['ticker']}")
    # Overlay SMA's
    import plotly.graph_objects as go
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_Fast"], name=f"SMA {sma_fast}", mode="lines"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_Slow"], name=f"SMA {sma_slow}", mode="lines"))
    st.plotly_chart(fig, use_container_width=True)

    if show_rsi:
        from plotly.subplots import make_subplots
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.7,0.3])
        fig2.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Prijs"), row=1, col=1)
        fig2.add_trace(go.Scatter(x=df["Date"], y=df["SMA_Fast"], name=f"SMA {sma_fast}", mode="lines"), row=1, col=1)
        fig2.add_trace(go.Scatter(x=df["Date"], y=df["SMA_Slow"], name=f"SMA {sma_slow}", mode="lines"), row=1, col=1)
        df["RSI14"] = rsi(df["Close"], 14)
        fig2.add_trace(go.Scatter(x=df["Date"], y=df["RSI14"], name="RSI(14)", mode="lines"), row=2, col=1)
        fig2.update_layout(template="plotly_white", height=560, xaxis_rangeslider_visible=False, title="Prijs + RSI")
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Geen koersdata gevonden voor deze combinatie van periode/interval.")

st.markdown("---")
st.subheader("⏱️ Snel overzicht watchlist")
if tickers:
    dfq = fetch_bulk_quotes(tickers)
    # Clean numbers for display
    dfq["price"] = dfq["price"].map(lambda x: round(float(x), 4) if x is not None else None)
    dfq["change"] = dfq["change"].map(lambda x: round(float(x), 4) if x is not None else None)
    dfq["change_pct"] = dfq["change_pct"].map(lambda x: round(float(x), 2) if x is not None else None)
    st.dataframe(dfq, use_container_width=True)
else:
    st.write("Voeg tickers toe in de sidebar.")
