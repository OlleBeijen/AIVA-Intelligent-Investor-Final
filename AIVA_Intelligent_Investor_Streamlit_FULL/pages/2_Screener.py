
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import numpy as np

# Optional direct source
try:
    import yfinance as yf
except Exception:
    yf = None

# Try imports from aiva_core; if missing modules exist, the offline mode still works
try:
    from aiva_core.data_sources import fetch_prices
except Exception:
    fetch_prices = None

from aiva_core.news import get_news
try:
    from aiva_core.advanced.conformal import calibrate_tau_precision, coverage, precision_at_mask
    from aiva_core.advanced.news_features import build_news_features
    from aiva_core.advanced.options_features import options_features
    from aiva_core.advanced.microstructure import intraday_features
    from aiva_core.advanced.regime_moe import regime_features, pick_expert
    from aiva_core.advanced.quantile_target import fit_quantiles, predict_quantiles, decision_from_quantiles
    from aiva_core.advanced.oof import time_series_oof_probs
    from aiva_core.advanced.triple_barrier import triple_barrier_labels, meta_label_from_direction
    from aiva_core.advanced.meta_label import train_meta_model
    from aiva_core.advanced.explain import permutation_importance
except Exception as e:
    st.warning(f"Advanced modules: {e}")

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score

st.set_page_config(page_title="Screener PRO – Offline Ready", page_icon="🧠", layout="wide")
st.title("🧠 Screener PRO (Offline Ready)")
st.caption("Werkt met live data (yfinance) of met meegeleverde sample-data als de netwerktoegang beperkt is.")

def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    # Title-casing kan multiindex/kolomnamen oplossen
    df.columns = [str(c).split('(')[0].strip().title() for c in df.columns]
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df["Close"] = df["Adj Close"]
    # Zorg voor basis kolommen
    for k in ["Open","High","Low","Close","Volume"]:
        if k not in df.columns:
            df[k] = np.nan
    if isinstance(df.index, pd.DatetimeIndex):
        df = df[~df.index.duplicated(keep="last")]
    return df

def load_from_sample(ticker: str) -> pd.DataFrame:
    p = Path(__file__).resolve().parents[1] / "sample_data" / f"{ticker}.csv"
    if not p.exists():
        # probeer ticker met vervangingen (punten/strepen)
        p = Path(__file__).resolve().parents[1] / "sample_data" / f"{ticker.replace('-', '_')}.csv"
    if p.exists():
        df = pd.read_csv(p, parse_dates=["Date"], index_col="Date")
        return _clean_cols(df)
    return pd.DataFrame()

def load_prices_robust(tickers, lookback_days: int, use_offline: bool, diag: dict):
    out = {}
    # 1) fetch_prices indien beschikbaar en online
    if fetch_prices is not None and not use_offline:
        try:
            data = fetch_prices(tickers, lookback_days=int(lookback_days))
        except Exception as e:
            data = {}
            diag["global"] = f"fetch_prices error: {e}"
    else:
        data = {}
        if use_offline:
            diag["global"] = "Offline modus AAN"
    # 2) per ticker beslissen
    for t in tickers:
        df = data.get(t) if isinstance(data, dict) else None
        if (df is None or len(df)==0) and not use_offline and yf is not None:
            # fallback yfinance direct
            try:
                df = yf.download(t, period=f"{int(lookback_days)}d", interval="1d", auto_adjust=True, progress=False)
            except Exception as e:
                diag[t] = f"yf fout: {e}"
                df = pd.DataFrame()
        if (df is None or df.empty) and use_offline:
            df = load_from_sample(t)
            if df.empty:
                # laatste redmiddel: pak AAPL sample voor alle
                df = load_from_sample("AAPL")
                diag[t] = diag.get(t, "") + " | fallback naar sample AAPL"
        df = _clean_cols(df)
        out[t] = df
        if t not in diag:
            diag[t] = f"{len(df)} rijen"
    return out

# ===== Controls =====
c1, c2, c3, c4 = st.columns([2,1,1,1])
with c1:
    tickers = st.text_input("Tickers", value="AAPL,MSFT,ASML.AS").strip()
    tickers = [t.strip() for t in tickers.split(",") if t.strip()]
with c2:
    lookback = st.number_input("Lookback (dagen)", 60, 1500, 365)
    horizon = st.number_input("Horizon (dagen)", 3, 30, 5)
with c3:
    eps = st.slider("ε (precision-filter)", 0.01, 0.3, 0.1, step=0.01)
with c4:
    offline = st.toggle("Offline modus", value=False, help="Gebruik sample-data als netwerk niet werkt.")

with st.expander("Geavanceerd"):
    use_news = st.checkbox("News features", value=True)
    use_options = st.checkbox("Options/IV features", value=True)
    build_portfolio = st.checkbox("Portefeuille optimaliseren", value=True)
    max_w = st.slider("Max gewicht per naam", 0.05, 0.5, 0.25, step=0.05)
    risk_averse = st.slider("Risico-parameter (λ)", 1.0, 10.0, 5.0, step=0.5)
    turn_pen = st.slider("Turnover-penalty (γ)", 0.0, 0.05, 0.01, step=0.005)
    max_part = st.slider("POV max participatie", 0.05, 0.5, 0.1, step=0.05)
    max_spread_bps = st.slider("Max spread (bps)", 5, 50, 15, step=1)
    default_qty = st.number_input("Stuks per naam", 1, 10000, 100)

run = st.button("Run Screener")
tabs = st.tabs(["📋 Selectie", "🧪 Diagnostics", "📈 Portefeuille", "🔍 Uitleg", "🧾 Export"])
tab_sel, tab_diag, tab_pf, tab_explain, tab_export = tabs

if run:
    diag = {}
    prices = load_prices_robust(tickers, int(lookback), offline, diag)

    # ===== Features =====
    rows, per_ticker_rows = [], {}
    for t, df in prices.items():
        if df is None or df.empty or "Close" not in df.columns:
            diag[t] = diag.get(t,"") + " | geen Close"
            continue
        c = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if len(c) < (int(horizon)+30):
            diag[t] = diag.get(t,"") + f" | te weinig bars: {len(c)} (min ~{int(horizon)+30})"
            continue
        r = c.pct_change().fillna(0.0)
        m = pd.DataFrame(index=c.index)
        m["ret1"] = r.shift(1)
        m["ret5"] = c.pct_change(5).shift(1)
        m["vol20"] = r.rolling(20).std().shift(1)
        m["mom20"] = r.rolling(20).mean().shift(1)
        m["bbw"] = (pd.to_numeric(df["High"], errors="coerce") - pd.to_numeric(df["Low"], errors="coerce")).rolling(20).mean().shift(1) / (c.shift(1)+1e-9)
        y = c.pct_change(int(horizon)).shift(-int(horizon)).reindex(m.index)
        m["ticker"] = t; m["target"] = y.values; m["y_bin"] = (y>0).astype(int).values
        keep = m.dropna()
        if keep.empty:
            diag[t] = diag.get(t,"") + " | na dropna leeg (kies lagere lookback/horizon)"
            continue
        rows.append(keep); per_ticker_rows[t] = len(keep)

    if not rows:
        with tab_diag:
            st.error("Geen bruikbare data — zie oorzaken hieronder.")
            st.dataframe(pd.DataFrame([{"ticker":k, "status":v} for k,v in diag.items()]))
        st.stop()

    feats = pd.concat(rows).reset_index().rename(columns={"index":"Date"})
    base_cols = ["ret1","ret5","vol20","mom20","bbw"]

    # ===== Model + conformal =====
    X = feats[base_cols]; y = feats["y_bin"].astype(int)
    n = len(X); cut = int(n*0.7)
    X_train, y_train = X.iloc[:cut], y.iloc[:cut]
    X_cal, y_cal = X.iloc[cut:], y.iloc[cut:]
    ctor = lambda: GradientBoostingClassifier(random_state=42)
    base = ctor(); base.fit(X_train, y_train)
    p_cal = base.predict_proba(X_cal)[:,1]
    tau = calibrate_tau_precision(y_cal.values, p_cal, eps=float(eps))

    # ===== OOF + TB meta =====
    from aiva_core.advanced.oof import time_series_oof_probs
    oof = time_series_oof_probs(ctor, X, y, n_splits=5)
    feats["p_oof"] = oof; feats["dir_sig"] = np.where(feats["p_oof"]>0.5, 1.0, -1.0)
    tb_all = []
    for t in feats["ticker"].unique():
        dfp = prices.get(t)
        if dfp is None or dfp.empty: continue
        tb = triple_barrier_labels(dfp["Close"], tp=0.04, sl=0.03, max_hold=int(horizon))
        sub = feats[feats["ticker"]==t]
        tb = tb.reindex(sub["Date"]).fillna(0.0)
        tb_all.append(pd.DataFrame({"idx": sub.index, "tb": tb.values}))
    if tb_all:
        tb_df = pd.concat(tb_all).set_index("idx").sort_index()
        feats["tb"] = tb_df["tb"].reindex(feats.index).fillna(0.0)
    else:
        feats["tb"] = 0.0
    meta_y = meta_label_from_direction(pd.Series(feats["dir_sig"], index=feats.index), pd.Series(feats["tb"], index=feats.index))
    meta_X = feats[base_cols]
    meta_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42).fit(meta_X, meta_y)

    # ===== Current snapshot =====
    last_rows = []
    for t in feats["ticker"].unique():
        sub = feats[feats["ticker"]==t]
        if len(sub) < 60:
            diag[t] = diag.get(t,"") + f" | recent te weinig rijen ({len(sub)})"
            continue
        last_rows.append(sub.iloc[[-1]][["ticker","Date"]+base_cols+["target","y_bin"]])
    if not last_rows:
        with tab_diag:
            st.error("Te weinig recente rijen per ticker na filtering.")
            st.dataframe(pd.DataFrame([{"ticker":k, "rows_after": per_ticker_rows.get(k,0), "status": diag.get(k,"")} for k in feats["ticker"].unique()]))
        st.stop()
    X_now = pd.concat(last_rows).set_index("ticker")

    # ===== Enrichment =====
    if use_news:
        items = get_news(tickers=list(X_now.index), limit_per=4, provider="auto")
        nf = build_news_features(items); X_now = X_now.join(nf.set_index("ticker"), on="ticker")
    else:
        X_now["claim_strength"] = 0.0; X_now["share_of_voice"] = 0.0
    if use_options:
        of = pd.DataFrame([{"ticker": t, **options_features(t)} for t in X_now.index]).set_index("ticker")
        X_now = X_now.join(of, on="ticker")
    else:
        X_now["iv_skew"] = np.nan; X_now["iv_level"] = np.nan; X_now["oi_surge"] = np.nan

    # microstructure optioneel via intraday; in offline modus kan dit leeg zijn
    try:
        mic = pd.DataFrame([{"ticker": t, **intraday_features(t)} for t in X_now.index]).set_index("ticker")
        X_now = X_now.join(mic, on="ticker")
    except Exception:
        X_now["range_contraction"] = np.nan; X_now["vwap_diff"] = np.nan; X_now["spread_proxy"] = np.nan

    # ===== Selectie =====
    p_now = base.predict_proba(X_now[base_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0))[:,1]
    sel = p_now > tau
    meta_now = meta_model.predict_proba(X_now[base_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0))[:,1] > 0.5
    q_models = fit_quantiles(X, feats["target"].astype(float))
    q_pred = predict_quantiles(q_models, X_now[base_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0))
    q_dec = decision_from_quantiles(q_pred, ret_thresh=0.04)
    take = sel & meta_now & q_dec.values

    results = X_now.copy()
    results["p_hat"] = p_now; results["meta_ok"] = meta_now
    results["q10"] = q_pred.get(0.1); results["q50"] = q_pred.get(0.5); results["q90"] = q_pred.get(0.9)
    results["take"] = take

    covg = coverage(p_now > tau); prec = precision_at_mask(y_cal.values, p_cal, tau)
    m1, m2, m3 = st.columns(3)
    m1.metric("τ (filter)", f"{tau:.3f}")
    m2.metric("Precision@τ (cal)", f"{prec:.2f}")
    m3.metric("Coverage@τ (nu)", f"{covg:.2f}")

    with tab_sel:
        st.subheader("Kandidaten")
        view = results[["Date","p_hat","meta_ok","q10","q50","q90","share_of_voice","claim_strength","iv_level","iv_skew","oi_surge","range_contraction","vwap_diff","spread_proxy","take"]].sort_values("p_hat", ascending=False)
        st.dataframe(view, use_container_width=True)

    with tab_diag:
        st.subheader("Diagnostics")
        rows = [{"ticker":k, "status":v} for k,v in diag.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with tab_pf:
        if build_portfolio and results["take"].any():
            from aiva_core.advanced.portfolio_opt import optimize_weights
            exp_ret = results.loc[results["take"], "q50"].rename("exp_ret")
            cov = feats.pivot_table(index="Date", columns="ticker", values="target").dropna().tail(120).cov()
            w = optimize_weights(exp_ret, cov, prev_w=None, risk_aversion=float(risk_averse), turn_penalty=float(turn_pen), max_w=float(max_w))
            st.subheader("Voorgestelde wegingen"); st.table(w.to_frame("weight"))
        else:
            st.info("Geen portefeuille gebouwd.")

    with tab_explain:
        try:
            imp = permutation_importance(base, X_train, y_train, metric=average_precision_score, n_repeats=5)
            st.subheader("Belangrijkste drivers"); st.write(imp.head(5))
        except Exception:
            st.write("—")

    with tab_export:
        out = results.reset_index().rename(columns={"index":"ticker"})
        st.download_button("Download resultaten (CSV)", out.to_csv(index=False).encode("utf-8"), "screener_pro_results.csv", "text/csv")
