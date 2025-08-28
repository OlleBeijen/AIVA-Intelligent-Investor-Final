
from __future__ import annotations
import numpy as np
import pandas as pd

def triple_barrier_labels(close: pd.Series, tp: float = 0.04, sl: float = 0.03, max_hold: int = 5) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce").ffill().dropna()
    n = len(c)
    out = pd.Series(index=c.index, dtype=float)
    for i in range(n):
        p0 = c.iloc[i]
        up = p0 * (1.0 + tp)
        dn = p0 * (1.0 - sl)
        end = min(n, i + 1 + int(max_hold))
        win = c.iloc[i+1:end]
        label = 0.0
        hit_up = (win >= up).any()
        hit_dn = (win <= dn).any()
        if hit_up and not hit_dn:
            label = 1.0
        elif hit_dn and not hit_up:
            label = -1.0
        else:
            j_up = np.where(win.values >= up)[0]
            j_dn = np.where(win.values <= dn)[0]
            if len(j_up) and len(j_dn):
                label = 1.0 if j_up[0] < j_dn[0] else -1.0
            else:
                label = 0.0
        out.iloc[i] = label
    return out

def meta_label_from_direction(signal_dir: pd.Series, tb_labels: pd.Series) -> pd.Series:
    s = pd.to_numeric(signal_dir, errors="coerce").reindex(tb_labels.index).fillna(0.0)
    tb = pd.to_numeric(tb_labels, errors="coerce").fillna(0.0)
    meta = ((s > 0) & (tb > 0)) | ((s < 0) & (tb < 0))
    return meta.astype(int)
