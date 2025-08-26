
import streamlit as st
from aiva_core.alerts import build_alerts
from aiva_core.data_sources import latest_close

st.set_page_config(page_title="Alerts – AIVA", page_icon="🔔", layout="wide")
st.title("🔔 Alerts (sessie-gebaseerd)")
st.caption("Controleer of prijzen over ingestelde drempels gaan. Draait alleen zolang de app open staat.")

tickers = st.text_input("Tickers", value="AAPL,MSFT,ASML.AS")
thresh = st.number_input("Drempel % (absolute verandering t.o.v. laatste close)", min_value=1, max_value=50, value=5)

if st.button("Check nu"):
    ts = [t.strip() for t in tickers.split(",") if t.strip()]
    prices = latest_close(ts, lookback_days=5)
    alerts = build_alerts(prices, threshold_pct=float(thresh))
    if not alerts:
        st.success("Geen alerts.")
    else:
        for a in alerts:
            st.warning(f"[{a.get('ticker')}] {a.get('message')}")
