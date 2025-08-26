
import streamlit as st
from aiva_core.agent import run_day
from aiva_core.utils import load_config, save_config

st.set_page_config(page_title="Agent – AIVA", page_icon="🕹️", layout="wide")
st.title("🕹️ Dagcyclus Agent")

st.caption("Bewerkt config en draait de dag-cyclus (data, signalen, forecast, sector-rapport, kansen).")

with st.expander("Config (config.yaml)"):
    try:
        cfg = load_config("config.yaml")
    except Exception:
        cfg = {"universe":["AAPL","MSFT","ASML.AS","AD.AS"], "risk":{"max_weight":0.2}}
    txt = st.text_area("config.yaml", value=__import__("yaml").safe_dump(cfg, sort_keys=False, allow_unicode=True), height=240)
    if st.button("Opslaan config"):
        try:
            obj = __import__("yaml").safe_load(txt)
            save_config(obj, "config.yaml")
            st.success("Opgeslagen.")
        except Exception as e:
            st.error(f"Fout: {e}")

if st.button("Run dagcyclus"):
    out = run_day("config.yaml")
    st.json(out)
