
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

def estimate_event_uplift(features: pd.DataFrame, y: pd.Series, treat: pd.Series) -> pd.Series:
    X = features.replace([np.inf,-np.inf], np.nan).fillna(0.0)
    y = pd.to_numeric(y, errors="coerce").fillna(0.0)
    t = (treat.astype(int) == 1)
    if t.sum() < 20 or (~t).sum() < 20:
        return pd.Series(index=features.index, dtype=float)
    model_t = GradientBoostingRegressor(random_state=42)
    model_c = GradientBoostingRegressor(random_state=42)
    model_t.fit(X[t], y[t])
    model_c.fit(X[~t], y[~t])
    uplift = pd.Series(model_t.predict(X) - model_c.predict(X), index=features.index)
    return uplift
