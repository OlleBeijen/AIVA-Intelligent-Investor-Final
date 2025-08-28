
from __future__ import annotations
from typing import Dict, List
import math, datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

def _estimate_intraday_profile(df: pd.DataFrame) -> pd.Series:
    """
    Schat deelname over de dag met eenvoudige volume-verdeling uit laatste sessie.
    Retourneert gewichten die optellen tot 1.
    """
    if df.empty or "Volume" not in df.columns:
        return pd.Series([1.0], index=[df.index[-1] if len(df) else dt.datetime.utcnow()])
    day = df.index[-1].date()
    ddf = df[df.index.date == day]
    if len(ddf) < 10:
        return pd.Series([1.0], index=[df.index[-1]])
    v = ddf["Volume"].replace(0, np.nan).fillna(method="ffill").fillna(1.0)
    w = v / (v.sum() + 1e-9)
    return w

def pov_schedule(total_qty: float, max_participation: float, profile: pd.Series) -> pd.DataFrame:
    """
    Bouw een POV schema: child-orders ≤ max_participation * (bar volume).
    We schalen zodanig dat de som ≈ total_qty.
    """
    w = profile / (profile.max() + 1e-9)  # schaal naar 0..1
    alloc = w / (w.sum() + 1e-9) * total_qty
    # respecteer max_participation: cap child-size -> we schalen omlaag en verdelen rest
    child = alloc.copy()
    cap = max(1e-9, float(max_participation))
    # geen echte bar-volume; neem proxy via profile*1000 om vorm te volgen
    bar_cap = profile * cap * 1000.0
    child = np.minimum(child.values, bar_cap.values)
    # normaliseer naar total_qty
    scale = total_qty / (child.sum() + 1e-9)
    child = child * scale
    return pd.DataFrame({"time": profile.index, "qty": child}, index=range(len(child)))

def throttle_spread(spread_proxy: float, max_spread_bps: float) -> float:
    """
    Geeft 0..1 throttle factor. Hogere spread → lagere factor.
    spread_proxy als fractie (bv 0.001 = 10bps).
    """
    if spread_proxy is None or not np.isfinite(spread_proxy):
        return 1.0
    bps = float(spread_proxy) * 10000.0
    if bps <= max_spread_bps:
        return 1.0
    # lineair aflopend tot 0 bij 3x max
    return max(0.0, 1.0 - (bps - max_spread_bps) / (2.0 * max_spread_bps))

def generate_execution_plan(ticker: str, side: str, target_qty: float, max_participation: float = 0.1,
                            max_spread_bps: float = 15.0) -> pd.DataFrame:
    """
    Simpel uitvoeringsplan: haal intraday bars, bouw POV schema, throttle op spread.
    """
    try:
        df = yf.download(ticker, period="5d", interval="5m", auto_adjust=True, progress=False)
    except Exception:
        df = pd.DataFrame()
    profile = _estimate_intraday_profile(df if isinstance(df, pd.DataFrame) else pd.DataFrame())
    plan = pov_schedule(abs(float(target_qty)), float(max_participation), profile)
    # spread proxy uit laatste bar (High-Low)/Close
    if isinstance(df, pd.DataFrame) and not df.empty:
        spr = float((df["High"].iloc[-1] - df["Low"].iloc[-1]) / (df["Close"].iloc[-1] + 1e-9))
    else:
        spr = math.nan
    factor = throttle_spread(spr, float(max_spread_bps))
    plan["qty"] = plan["qty"] * factor
    plan["side"] = side.upper()
    plan["ticker"] = ticker
    plan["limit_hint"] = "bid+0.2*spr" if side.lower()=="buy" else "ask-0.2*spr"
    return plan
