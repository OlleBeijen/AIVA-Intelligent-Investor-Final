
from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

def train_meta_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict[str, float]]:
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = y.astype(int)
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    clf.fit(X, y)
    try:
        auc = float(roc_auc_score(y, clf.predict_proba(X)[:,1]))
    except Exception:
        auc = float("nan")
    return clf, {"auc_train": auc}
