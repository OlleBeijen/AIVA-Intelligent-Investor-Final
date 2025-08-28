
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

def fit_quantiles(X: pd.DataFrame, y: pd.Series, qs=(0.1,0.5,0.9)):
    models = {}
    X = X.replace([np.inf,-np.inf], np.nan).fillna(0.0)
    y = pd.to_numeric(y, errors="coerce").fillna(0.0)
    for q in qs:
        m = QuantileRegressor(quantile=q, alpha=0.0001)
        m.fit(X, y)
        models[q] = m
    return models

def predict_quantiles(models, X: pd.DataFrame) -> pd.DataFrame:
    X = X.replace([np.inf,-np.inf], np.nan).fillna(0.0)
    out = {}
    for q, m in models.items():
        out[q] = m.predict(X)
    return pd.DataFrame(out, index=X.index)

def decision_from_quantiles(qdf: pd.DataFrame, ret_thresh: float = 0.04) -> pd.Series:
    q10 = qdf.get(0.1); q50 = qdf.get(0.5)
    if q10 is None or q50 is None:
        return pd.Series([False]*len(qdf), index=qdf.index)
    return (q50 >= ret_thresh) & (q10 > -ret_thresh/2.0)
