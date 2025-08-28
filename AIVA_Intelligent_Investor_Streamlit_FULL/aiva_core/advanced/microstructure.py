
from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf

def intraday_features(ticker: str, period: str = "5d", interval: str = "5m") -> dict:
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    except Exception:
        return {"range_contraction": float("nan"), "vwap_diff": float("nan"), "spread_proxy": float("nan")}
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"range_contraction": float("nan"), "vwap_diff": float("nan"), "spread_proxy": float("nan")}
    df = df.dropna()
    df["hl"] = df["High"] - df["Low"]
    atr = df["hl"].rolling(20).mean()
    rc = float(atr.iloc[-1] / (atr.rolling(60).mean().iloc[-1] + 1e-9)) if len(atr) > 60 else float("nan")
    day = df.index[-1].date()
    day_df = df[df.index.date == day]
    if len(day_df):
        vwap = float((day_df["Close"] * day_df["Volume"]).sum() / (day_df["Volume"].sum() + 1e-9))
        vwap_diff = float((day_df["Close"].iloc[-1] - vwap) / (vwap + 1e-9))
    else:
        vwap_diff = float("nan")
    try:
        spr = float((df["High"].iloc[-1] - df["Low"].iloc[-1]) / (df["Close"].iloc[-1] + 1e-9))
    except Exception:
        spr = float("nan")
    return {"range_contraction": rc, "vwap_diff": vwap_diff, "spread_proxy": spr}
