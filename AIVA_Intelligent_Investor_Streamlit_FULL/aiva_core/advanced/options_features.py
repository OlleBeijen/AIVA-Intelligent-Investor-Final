
from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
import yfinance as yf

def _near_term_chain(ticker: str):
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        if not exps: return None, None
        exp = exps[0]
        oc = t.option_chain(exp)
        return exp, (oc.calls, oc.puts)
    except Exception:
        return None, (pd.DataFrame(), pd.DataFrame())

def _iv_skew_approx(calls: pd.DataFrame, puts: pd.DataFrame) -> float:
    try:
        k_med = float(np.median(pd.concat([calls["strike"], puts["strike"]]).values))
        call_mask = (calls["strike"] >= 1.05*k_med)
        put_mask  = (puts["strike"]  <= 0.95*k_med)
        ivc = float(calls.loc[call_mask, "impliedVolatility"].median())
        ivp = float(puts.loc[put_mask,  "impliedVolatility"].median())
        return ivp - ivc
    except Exception:
        return float("nan")

def options_features(ticker: str) -> Dict[str, Any]:
    exp, chain = _near_term_chain(ticker)
    if not exp or chain is None:
        return {"iv_skew": float("nan"), "iv_level": float("nan"), "oi_surge": float("nan")}
    calls, puts = chain
    try:
        iv_level = float(pd.concat([calls["impliedVolatility"], puts["impliedVolatility"]]).median())
    except Exception:
        iv_level = float("nan")
    skew = _iv_skew_approx(calls, puts)
    try:
        oi = pd.concat([calls["openInterest"], puts["openInterest"]]).fillna(0)
        oi_surge = float(oi.quantile(0.9) / (oi.quantile(0.5) + 1e-6))
    except Exception:
        oi_surge = float("nan")
    return {"iv_skew": skew, "iv_level": iv_level, "oi_surge": oi_surge}
