
from __future__ import annotations
import numpy as np
import pandas as pd

def regime_features(mkt: pd.Series) -> dict:
    r = pd.to_numeric(mkt, errors="coerce").pct_change().dropna()
    vol = float(r.rolling(20).std().iloc[-1]) if len(r) >= 20 else float("nan")
    trend = float((r.rolling(20).mean().iloc[-1])) if len(r) >= 20 else float("nan")
    return {"mkt_vol20": vol, "mkt_trend20": trend}

def pick_expert(reg_feats: dict) -> str:
    v = reg_feats.get("mkt_vol20")
    t = reg_feats.get("mkt_trend20")
    if isinstance(v, float) and v > 0.02:
        return "momentum_breakout"
    if isinstance(t, float) and t > 0:
        return "news_drift"
    return "value_rerate"
