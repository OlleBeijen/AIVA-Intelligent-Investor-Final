
# AIVA – Intelligent Investor (Streamlit)

Een frisse, Nederlandstalige beleggingsapp die draait op **Streamlit** met:
- Live koersen (aandelen, indexen, crypto via `yfinance`)
- Overzicht van markten
- Simpele **screener**
- **Portefeuille**-tracking met P&L
- **Backtest** (SMA crossover)
- **AI Analist** (optioneel – OpenAI)

## Snel starten

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## AI Analist (OpenAI, optioneel)
Zet je API key in `.streamlit/secrets.toml`:
```toml
OPENAI_API_KEY="sk-..."
```

of als omgevingsvariabele:
```bash
export OPENAI_API_KEY="sk-..."
```

## Tips
- Voor crypto: gebruik tickers zoals `BTC-USD`, `ETH-USD`.
- Voor Europese aandelen via Yahoo: `ASML.AS`, `AD.AS`, `PHIA.AS`.
- De app gebruikt caching en ververst bij elke actie. Gebruik lagere intervallen (1m/5m) met korte periodes.

## Disclaimer
Dit is geen financieel advies. Doe altijd je eigen onderzoek.


## API keys (optioneel)

Voor uitgebreid nieuws en LLM-chat kun je de volgende omgevingsvariabelen zetten:
- `NEWSAPI_KEY` (NewsAPI.org)
- `POLYGON_KEY` (Polygon.io)
- `FINNHUB_KEY` (Finnhub.io)
- `FMP_KEY` (FinancialModelingPrep)
- `OPENAI_API_KEY` (OpenAI)
- `GROQ_API_KEY` (Groq) en `LLM_PROVIDER="groq"`
- `GEMINI_API_KEY` (Google) en `LLM_PROVIDER="gemini"`
- `HF_API_TOKEN` (HuggingFace) en `LLM_PROVIDER="hf"`

Zet ze in `.streamlit/secrets.toml` of als env-vars.
