
from __future__ import annotations
from typing import Dict, Any
import os, requests

def crowding_from_fmp(ticker: str) -> Dict[str, float]:
    key = os.getenv("FMP_KEY")
    out = {"short_interest": float("nan"), "days_to_cover": float("nan"), "borrow_fee": float("nan")}
    if not key:
        return out
    base = "https://financialmodelingprep.com/api/v4/"
    try:
        r = requests.get(base + "short_interest", params={"symbol": ticker, "apikey": key}, timeout=15)
        if r.ok and r.json():
            it = r.json()[0]
            out["short_interest"] = float(it.get("shortInterest", float("nan")))
            out["days_to_cover"] = float(it.get("daysToCover", float("nan")))
    except Exception:
        pass
    return out
