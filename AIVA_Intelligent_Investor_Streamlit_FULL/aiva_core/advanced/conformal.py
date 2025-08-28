
from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

def calibrate_tau_precision(y_cal: np.ndarray, p_cal: np.ndarray, eps: float = 0.1) -> float:
    y_cal = np.asarray(y_cal).astype(int)
    p_cal = np.asarray(p_cal).astype(float)
    if len(y_cal) != len(p_cal) or len(y_cal) == 0:
        return 0.999
    uniq = np.unique(p_cal)
    uniq.sort(); uniq = uniq[::-1]
    target = 1.0 - float(eps)
    best_tau = 0.999
    for tau in uniq:
        mask = p_cal > tau
        if not np.any(mask):
            continue
        prec = float((y_cal[mask] == 1).mean())
        if np.isfinite(prec) and prec >= target:
            best_tau = float(tau)
            break
    return float(best_tau)

def selective_mask(p: np.ndarray, tau: float) -> np.ndarray:
    return np.asarray(p) > float(tau)

def coverage(mask: np.ndarray) -> float:
    m = np.asarray(mask).astype(bool)
    return float(np.mean(m)) if len(m) else 0.0

def precision_at_mask(y: np.ndarray, p: np.ndarray, tau: float) -> float:
    m = selective_mask(p, tau)
    if not np.any(m):
        return float("nan")
    return float((np.asarray(y)[m] == 1).mean())
