
from __future__ import annotations
from typing import List, Dict, Any
import re, collections
import pandas as pd

_PATTERNS = [
    (r"\brais(e|es|ed|ing).*(guidance|outlook|revenue|sales|eps)", +2),
    (r"\bprice (increase|hike|rise)\b", +2),
    (r"\bfda\b.*\bphase\s*3\b.*(met|meet|primary|endpoint)", +3),
    (r"\bguidance\b.*(up|higher|above)", +2),
    (r"\bdowngrade|cut guidance|miss\b", -2),
    (r"\brecall\b|\bfraud\b|\bprobe\b|\bsec\b|\bantitrust\b", -3),
]

def extract_claim_strength(text: str) -> int:
    s = (text or "").lower()
    score = 0
    for pat, w in _PATTERNS:
        if re.search(pat, s):
            score += w
    return score

def share_of_voice(items: List[Dict[str, Any]]) -> Dict[str, float]:
    by_day = collections.defaultdict(list)
    for it in items:
        day = (it.get("publishedAt","") or "")[:10]
        by_day[day].append(it)
    sov = {}
    for day, lst in by_day.items():
        total = max(1, len(lst))
        per_t = collections.Counter([it.get("ticker","") for it in lst])
        for t, c in per_t.items():
            sov[(day, t)] = float(c)/float(total)
    latest = {}
    if sov:
        last_day = sorted({d for d,_ in sov.keys()})[-1]
        for (d, t), v in sov.items():
            if d == last_day:
                latest[t] = v
    return latest

def build_news_features(items: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for it in items:
        t = it.get("ticker","")
        title = it.get("title","")
        s = extract_claim_strength(title)
        rows.append({"ticker": t, "claim_strength": s})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["ticker","claim_strength","share_of_voice"])
    sov = share_of_voice(items)
    df["share_of_voice"] = df["ticker"].map(sov).fillna(0.0)
    agg = df.groupby("ticker", as_index=False).agg({"claim_strength":"sum","share_of_voice":"max"})
    return agg
