
import streamlit as st
from aiva_core.news import get_news

st.set_page_config(page_title="Nieuws – AIVA", page_icon="🗞️", layout="wide")
st.title("🗞️ Nieuws & Headlines")
st.caption("Kies een bron en geef tickers op. Voor sommige bronnen heb je een API key nodig (zie README).")

provider = st.selectbox("Bron", ["auto","newsapi","finnhub","polygon","fmp"], index=0, help="auto probeert beschikbare keys.")

tickers = st.text_input("Tickers (kommagescheiden)", value="AAPL,MSFT,ASML.AS,BTC-USD")
limit = st.slider("Max per ticker", 1, 20, 8)

if st.button("Haal nieuws op"):
    ts = [t.strip() for t in tickers.split(",") if t.strip()]
    with st.spinner("Laden..."):
        items = get_news(tickers=ts, limit_per=limit, provider=provider)
    if not items:
        st.info("Geen nieuws gevonden (controleer API keys of probeer een andere bron).")
    else:
        for it in items:
            with st.container(border=True):
                st.markdown(f"**[{it.get('ticker','')}] {it.get('title','')}**")
                st.write(it.get("publisher",""))
                link = it.get("link","")
                if link:
                    st.write(f"[Open artikel]({link})")
                st.caption(f"Published: {it.get('publishedAt','')}")
