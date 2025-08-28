
from __future__ import annotations
import numpy as np
import pandas as pd

def psi(a: pd.Series, b: pd.Series, bins: int = 10) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 10 or len(b) < 10: return float("nan")
    cuts = np.quantile(a, np.linspace(0,1,bins+1))
    cuts[0] = -np.inf; cuts[-1] = np.inf
    pa, _ = np.histogram(a, bins=cuts); pb, _ = np.histogram(b, bins=cuts)
    pa = pa / (pa.sum()+1e-9); pb = pb / (pb.sum()+1e-9)
    out = np.sum((pa - pb) * np.log((pa + 1e-9)/(pb + 1e-9)))
    return float(out)
