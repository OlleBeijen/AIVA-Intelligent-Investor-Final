
import streamlit as st
import pandas as pd
from utils.data import fetch_history
from utils.indicators import sma

st.set_page_config(page_title="Backtest – AIVA", page_icon="🧪", layout="wide")

st.title("🧪 Backtest – SMA Crossover")
st.caption("Eenvoudige long-only strategie: koop wanneer SMA(kort) > SMA(lang), anders cash.")

t = st.text_input("Ticker", value="AAPL")
period = st.selectbox("Periode", ["2y","5y","10y","max"], index=2)
fast = st.number_input("SMA kort", min_value=5, max_value=200, value=20, step=1)
slow = st.number_input("SMA lang", min_value=10, max_value=400, value=50, step=1)

df = fetch_history(t, period=period, interval="1d")
if df.empty:
    st.info("Geen data.")
    st.stop()

df["SMA_F"] = sma(df["Close"], fast)
df["SMA_S"] = sma(df["Close"], slow)
df["Signal"] = (df["SMA_F"] > df["SMA_S"]).astype(int)
df["Return"] = df["Close"].pct_change().fillna(0.0)
df["Strat"] = df["Signal"].shift(1).fillna(0) * df["Return"]
df["CumBuyHold"] = (1 + df["Return"]).cumprod()
df["CumStrat"] = (1 + df["Strat"]).cumprod()

import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"], y=df["CumBuyHold"], mode="lines", name="Buy & Hold"))
fig.add_trace(go.Scatter(x=df["Date"], y=df["CumStrat"], mode="lines", name="SMA Strategie"))
fig.update_layout(template="plotly_white", title="Groei van €1", height=420)
st.plotly_chart(fig, use_container_width=True)

# Statistieken
def max_drawdown(series: pd.Series) -> float:
    roll_max = series.cummax()
    dd = series / roll_max - 1.0
    return dd.min()

stats = {
    "Eindwaarde B&H": df["CumBuyHold"].iloc[-1],
    "Eindwaarde Strat": df["CumStrat"].iloc[-1],
    "CAGR B&H": (df["CumBuyHold"].iloc[-1] ** (252/len(df)) - 1) if len(df) > 252 else None,
    "CAGR Strat": (df["CumStrat"].iloc[-1] ** (252/len(df)) - 1) if len(df) > 252 else None,
    "Max DD B&H": max_drawdown(df["CumBuyHold"]),
    "Max DD Strat": max_drawdown(df["CumStrat"]),
    "Trades (approx)": int((df["Signal"].diff().abs() == 1).sum())
}
st.subheader("Statistieken")
st.json({k: (round(v,4) if isinstance(v, (int,float)) and v is not None else v) for k,v in stats.items()})
