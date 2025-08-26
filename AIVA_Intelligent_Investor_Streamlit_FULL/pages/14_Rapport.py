
import streamlit as st
import pandas as pd
from aiva_core.data_sources import latest_close
from aiva_core.utils import now_ams

st.set_page_config(page_title="Rapport – AIVA", page_icon="📝", layout="wide")
st.title("📝 Rapport")
st.caption("Genereer een kort Markdown-rapport met recente prijzen en opmerkingen.")

tickers = st.text_input("Tickers", value="AAPL,MSFT,ASML.AS")
notes = st.text_area("Opmerkingen", value="")

if st.button("Genereer"):
    ts = [t.strip() for t in tickers.split(",") if t.strip()]
    px = latest_close(ts, lookback_days=5)
    lines = [f"# AIVA Rapport – {now_ams()}", ""]
    for t, p in px.items():
        lines.append(f"- **{t}**: {p}")
    if notes:
        lines += ["", "## Notities", notes]
    md = "\n".join(lines)
    st.download_button("Download rapport.md", md.encode("utf-8"), "rapport.md", "text/markdown")
    st.markdown(md)
