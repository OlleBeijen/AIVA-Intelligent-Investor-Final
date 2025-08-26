
import os
import streamlit as st
from utils.data import fetch_quote, fetch_history
from utils.indicators import sma, rsi

st.set_page_config(page_title="AI Analist – AIVA", page_icon="🧠", layout="wide")

st.title("🧠 AI Analist")
st.caption("Krijg beknopte uitleg in gewone taal. Met of zonder OpenAI API-sleutel.")

t = st.text_input("Ticker", value="AAPL")
period = st.selectbox("Periode", ["6mo","1y","2y","5y"], index=1)

df = fetch_history(t, period=period, interval="1d")
q = fetch_quote(t)

if df.empty or q["price"] is None:
    st.info("Geen data beschikbaar voor analyse.")
    st.stop()

df["SMA20"] = sma(df["Close"], 20)
df["SMA50"] = sma(df["Close"], 50)
df["RSI14"] = rsi(df["Close"], 14)

# Samenvatting zonder LLM (regelgebaseerd)
last = df.iloc[-1]
summary_points = []
if last["SMA20"] and last["SMA50"]:
    if last["SMA20"] > last["SMA50"]:
        summary_points.append("Korte trend is opwaarts (SMA20 boven SMA50).")
    else:
        summary_points.append("Korte trend is neerwaarts (SMA20 onder SMA50).")
if last["RSI14"]:
    if last["RSI14"] > 70:
        summary_points.append("RSI wijst op mogelijke overbought-conditie (>70).")
    elif last["RSI14"] < 30:
        summary_points.append("RSI wijst op mogelijke oversold-conditie (<30).")
    else:
        summary_points.append("RSI is neutraal.")

summary_points.append(f"Huidige prijs: {q['price']} {q['currency']} (t: {q['ticker']}).")

st.subheader("Snelle uitleg (zonder AI)")
st.write("\n".join([f"- {p}" for p in summary_points]))

st.markdown("---")
st.subheader("Chat met AI (optioneel)")
st.caption("Zet je **OPENAI_API_KEY** in `.streamlit/secrets.toml` of als env-var.")

use_ai = st.toggle("OpenAI gebruiken", value=False)
if use_ai:
    api_key = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Geen OpenAI API-sleutel gevonden.")
    else:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = st.text_area("Vraag aan de analist", value=f"Geef een korte uitleg van {t} op basis van SMA(20/50), RSI(14) en recente koersbewegingen, in simpele Nederlandse taal.")
        if st.button("Vraag AI"):
            try:
                msg = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role":"system","content":"Je bent een nuchtere beleggingsanalist. Vermijd hype en vakjargon. Geef duidelijke, korte zinnen."},
                        {"role":"user","content": prompt}
                    ],
                    temperature=0.2,
                )
                st.write(msg.choices[0].message.content)
            except Exception as e:
                st.error(f"Fout bij AI-aanroep: {e}")
