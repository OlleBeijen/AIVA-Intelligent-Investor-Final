
import streamlit as st
import pandas as pd
from aiva_core.fundamentals import get_fundamentals

st.set_page_config(page_title="Fundamentals – AIVA", page_icon="📚", layout="wide")
st.title("📚 Fundamentals")
st.caption("Data via yfinance en optioneel FMP API.")

tickers = st.text_input("Tickers (kommagescheiden)", value="AAPL,MSFT,ASML.AS")
if st.button("Ophalen"):
    ts = [t.strip() for t in tickers.split(",") if t.strip()]
    data = get_fundamentals(ts)
    if not data:
        st.info("Geen data gevonden.")
    else:
        rows = []
        for t, d in data.items():
            rows.append({
                "ticker": t,
                "name": d.get("longName") or t,
                "price": d.get("price"),
                "marketCap": d.get("marketCap") or d.get("mktCap"),
                "currency": d.get("currency"),
                "pe": d.get("trailingPE") or d.get("pe"),
                "pb": d.get("priceToBook") or d.get("pb"),
                "evEbitda": d.get("enterpriseToEbitda") or d.get("ev_ebitda"),
                "fcf": d.get("freeCashflow") or d.get("fcf"),
                "link": d.get("link")
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
