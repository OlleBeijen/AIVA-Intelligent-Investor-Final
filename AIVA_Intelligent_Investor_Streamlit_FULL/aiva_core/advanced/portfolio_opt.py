
from __future__ import annotations
import numpy as np
import pandas as pd

def optimize_weights(exp_rets: pd.Series, cov: pd.DataFrame, prev_w: pd.Series | None = None,
                     risk_aversion: float = 5.0, turn_penalty: float = 0.01, max_w: float = 0.2) -> pd.Series:
    idx = exp_rets.index
    mu = exp_rets.fillna(0.0).values
    S = cov.reindex(index=idx, columns=idx).fillna(0.0).values
    n = len(mu)
    w = np.zeros(n) if prev_w is None else prev_w.reindex(idx).fillna(0.0).values.copy()
    def proj(w):
        w = np.clip(w, 0.0, max_w)
        s = w.sum()
        if s > 1.0: w *= (1.0 / s)
        return w
    lr = 0.1
    for _ in range(300):
        grad = mu - 2*risk_aversion*(S @ w)
        d = w - (prev_w.reindex(idx).fillna(0.0).values if prev_w is not None else 0.0)
        grad -= turn_penalty * np.sign(d)
        w = proj(w + lr*grad)
    return pd.Series(w, index=idx)
