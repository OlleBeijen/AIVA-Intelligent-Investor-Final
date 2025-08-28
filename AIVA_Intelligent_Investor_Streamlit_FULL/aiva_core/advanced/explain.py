
from __future__ import annotations
import numpy as np
import pandas as pd

def permutation_importance(model, X: pd.DataFrame, y: pd.Series, metric=None, n_repeats: int = 5):
    rng = np.random.default_rng(42)
    base = metric(y, model.predict_proba(X)[:,1]) if metric else None
    out = {}
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            val = metric(y, model.predict_proba(Xp)[:,1]) if metric else None
            if base is not None and val is not None:
                drops.append(base - val)
        out[col] = float(np.mean(drops)) if drops else float("nan")
    s = pd.Series(out).sort_values(ascending=False)
    return s
