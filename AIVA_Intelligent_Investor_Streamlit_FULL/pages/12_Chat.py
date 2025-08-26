
import streamlit as st
from aiva_core.chat import chat_answer

st.set_page_config(page_title="Chat – AIVA", page_icon="💬", layout="wide")
st.title("💬 Chat met AIVA (LLM)")
st.caption("Ondersteunt OpenAI, Groq, Gemini, HuggingFace via env-vars.")

prov = st.selectbox("Provider", ["openai","groq","gemini","hf"], index=0, help="Stel env variabelen in; zie README.")
msg = st.text_area("Jouw vraag", value="Wat vind je van AAPL en ASML?")
if st.button("Vraag"):
    import os
    os.environ["LLM_PROVIDER"] = prov
    out = chat_answer(msg, provider=prov)
    st.write(out.get("text","(geen antwoord)"))
    if out.get("tickers"):
        st.caption("Gevonden tickers: " + ", ".join(out["tickers"]))
