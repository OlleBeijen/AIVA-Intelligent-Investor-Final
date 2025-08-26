
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import yfinance as yf

DEFAULT_TZ = ZoneInfo("Europe/Amsterdam")

def _normalize_ticker(ticker: str) -> str:
    t = (ticker or "").strip().upper()
    # Allow shorthand for crypto (e.g., BTC => BTC-USD)
    if t and "-" not in t and t.isalpha() and len(t) in (3,4):
        # Try crypto guess first
        if t in {"BTC","ETH","SOL","ADA","DOGE","XRP","BNB","DOT","MATIC"}:
            return f"{t}-USD"
    return t

def fetch_quote(ticker: str) -> dict:
    """
    Fetch a light-weight live quote using yfinance fast_info, with fallbacks.
    Returns a dict with price, change, change_pct, currency, market_state, time.
    """
    t = _normalize_ticker(ticker)
    tk = yf.Ticker(t)
    now = datetime.now(tz=DEFAULT_TZ)

    price = None
    currency = None
    change = None
    change_pct = None
    market_state = "unknown"

    # Try fast_info first
    try:
        fi = tk.fast_info
        price = float(fi["last_price"])
        prev = float(fi.get("previous_close") or np.nan)
        currency = fi.get("currency") or "USD"
        if prev and prev == prev and prev != 0:
            change = price - prev
            change_pct = (change / prev) * 100.0
        market_state = fi.get("market_state") or "unknown"
    except Exception:
        pass

    # Fallback: use recent history for last close / last price
    if price is None:
        try:
            hist = tk.history(period="1d", interval="1m")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
                prev_close = float(hist["Close"].iloc[0])
                if prev_close:
                    change = price - prev_close
                    change_pct = (change / prev_close) * 100.0
            if currency is None:
                info = tk.info
                currency = info.get("currency", "USD")
        except Exception:
            pass

    return {
        "ticker": t,
        "price": price,
        "change": change,
        "change_pct": change_pct,
        "currency": currency or "USD",
        "market_state": market_state,
        "time": now.isoformat(),
    }

def fetch_history(ticker: str, period="1y", interval="1d") -> pd.DataFrame:
    t = _normalize_ticker(ticker)
    df = yf.download(t, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df.reset_index().rename(columns=str.title)
        # Ensure datetime with tz for safety
        if "Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"], utc=True)
        return df
    return pd.DataFrame(columns=["Date","Open","High","Low","Close","Adj Close","Volume"])

def fetch_bulk_quotes(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            q = fetch_quote(t)
            rows.append(q)
        except Exception:
            rows.append({"ticker": t, "price": None, "change": None, "change_pct": None, "currency":"", "market_state":"error", "time": datetime.now(tz=DEFAULT_TZ).isoformat()})
    return pd.DataFrame(rows)

def safe_number(x, d=2):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return round(float(x), d)
    except Exception:
        return None
