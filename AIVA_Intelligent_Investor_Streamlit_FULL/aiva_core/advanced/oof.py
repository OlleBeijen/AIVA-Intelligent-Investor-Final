
from __future__ import annotations
from typing import Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

def time_series_oof_probs(model_ctor, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, fit_params: dict | None = None) -> np.ndarray:
    """
    Genereer OOF-probabilities via time-series splits (expanding window).
    model_ctor: functie die een NIET-GETRAIND model teruggeeft (bv. lambda: GradientBoostingClassifier())
    Geeft probs[:,1] in volgorde van X/y.
    """
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True).astype(int)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = np.full(len(X), np.nan, dtype=float)
    for train_idx, test_idx in tscv.split(X):
        if len(train_idx) < 50:
            continue
        m = model_ctor()
        if fit_params:
            m.fit(X.iloc[train_idx], y.iloc[train_idx], **fit_params)
        else:
            m.fit(X.iloc[train_idx], y.iloc[train_idx])
        try:
            proba = m.predict_proba(X.iloc[test_idx])[:,1]
        except Exception:
            proba = m.decision_function(X.iloc[test_idx])
            proba = 1/(1+np.exp(-proba))
        oof[test_idx] = proba
    # backfill any NaNs with in-sample quick fit (last resort)
    if np.isnan(oof).any():
        m = model_ctor()
        m.fit(X, y)
        pred_all = m.predict_proba(X)[:,1] if hasattr(m, "predict_proba") else 1/(1+np.exp(-m.decision_function(X)))
        oof = np.where(np.isnan(oof), pred_all, oof)
    return oof
