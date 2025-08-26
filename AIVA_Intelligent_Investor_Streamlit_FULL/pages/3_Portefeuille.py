
import streamlit as st
import pandas as pd
from utils.data import fetch_quote

st.set_page_config(page_title="Portefeuille – AIVA", page_icon="💼", layout="wide")

st.title("💼 Portefeuille")
st.caption("Voeg posities toe en volg P&L. Je kunt ook een CSV importeren/exporteren.")

template = pd.DataFrame([
    {"Ticker": "AAPL", "Aantal": 10, "Inkoopprijs": 150.0, "Valuta": "USD"},
    {"Ticker": "ASML.AS", "Aantal": 5, "Inkoopprijs": 700.0, "Valuta": "EUR"}
])

if "pf" not in st.session_state:
    st.session_state.pf = template.copy()

uploaded = st.file_uploader("Importeer CSV (kolommen: Ticker, Aantal, Inkoopprijs, Valuta)", type=["csv"])
if uploaded:
    try:
        st.session_state.pf = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Kon CSV niet lezen: {e}")

st.data_editor(st.session_state.pf, num_rows="dynamic", use_container_width=True, key="pf_editor")

# Herbereken
rows = []
for _, r in st.session_state.pf.iterrows():
    t = r.get("Ticker")
    if not t or pd.isna(t):
        continue
    try:
        q = fetch_quote(t)
        prijs = q["price"]
        aantal = float(r.get("Aantal", 0) or 0)
        kost = float(r.get("Inkoopprijs", 0) or 0)
        waarde = (prijs or 0) * aantal
        inzet = kost * aantal
        p_l = waarde - inzet
        p_l_pct = (p_l / inzet * 100.0) if inzet else None
        rows.append({
            "Ticker": q["ticker"],
            "Aantal": aantal,
            "Gem. inkoop": kost,
            "Huidige prijs": prijs,
            "Waarde": waarde,
            "P/L": p_l,
            "P/L %": p_l_pct,
            "Valuta": q["currency"]
        })
    except Exception as e:
        rows.append({"Ticker": t, "Aantal": r.get("Aantal"), "Fout": str(e)})

df = pd.DataFrame(rows)
st.subheader("Overzicht")
st.dataframe(df, use_container_width=True)

totale_waarde = df["Waarde"].sum() if "Waarde" in df.columns else 0.0
totale_inzet = (df["Gem. inkoop"] * df["Aantal"]).sum() if "Gem. inkoop" in df.columns else 0.0
totaal_pl = totale_waarde - totale_inzet
totaal_pl_pct = (totaal_pl / totale_inzet * 100.0) if totale_inzet else None

col = st.columns(3)
col[0].metric("Totale waarde", f"{totale_waarde:,.2f}")
col[1].metric("Totaal P/L", f"{totaal_pl:,.2f}")
col[2].metric("Totaal P/L %", f"{totaal_pl_pct:,.2f}%" if totaal_pl_pct is not None else "n.v.t.")

st.download_button("Exporteer portefeuille (CSV)", st.session_state.pf.to_csv(index=False).encode("utf-8"), file_name="portefeuille.csv", mime="text/csv")
