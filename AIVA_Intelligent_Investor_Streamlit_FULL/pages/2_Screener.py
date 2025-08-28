
import streamlit as st
import pandas as pd
import numpy as np

from aiva_core.data_sources import fetch_prices
from aiva_core.news import get_news
from aiva_core.advanced.conformal import calibrate_tau_precision, coverage, precision_at_mask
from aiva_core.advanced.news_features import build_news_features
from aiva_core.advanced.options_features import options_features
from aiva_core.advanced.microstructure import intraday_features
from aiva_core.advanced.regime_moe import regime_features, pick_expert
from aiva_core.advanced.quantile_target import fit_quantiles, predict_quantiles, decision_from_quantiles
from aiva_core.advanced.portfolio_opt import optimize_weights
from aiva_core.advanced.explain import permutation_importance
from aiva_core.advanced.oof import time_series_oof_probs
from aiva_core.advanced.triple_barrier import triple_barrier_labels, meta_label_from_direction
from aiva_core.execution import generate_execution_plan

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score

st.set_page_config(page_title="Screener PRO – AIVA", page_icon="🧠", layout="wide")

# ====== UI Header ======
st.markdown(
    """
    <style>
    .smallcaps {font-variant: all-small-caps; letter-spacing: .06em; color:#6b7280;}
    .pill {padding:.2rem .5rem; border-radius: 999px; background:#eef2ff; color:#4338ca; margin-left:.5rem;}
    .good {color:#059669; font-weight:600;}
    .warn {color:#b45309; font-weight:600;}
    .bad  {color:#b91c1c; font-weight:600;}
    </style>
    """,
    unsafe_allow_html=True
)
st.title("🧠 Screener PRO")
st.caption("Selectief instappen met **conformal filtering** en **meta-labeling**. Extra signalen: nieuws, opties/IV, microstructure, regime-gating en quantile-doel.")

# ====== Controls ======
with st.container():
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        tickers = st.text_input("Tickers", value="AAPL,MSFT,AMZN,GOOGL,TSLA,ASML.AS,AD.AS")
        tickers = [t.strip() for t in tickers.split(",") if t.strip()]
    with c2:
        lookback = st.number_input("Lookback (dagen)", 180, 1500, 365)
        horizon = st.number_input("Horizon (dagen)", 3, 30, 5)
    with c3:
        eps = st.slider("ε (precision-filter)", 0.01, 0.3, 0.1, step=0.01)

with st.expander("Geavanceerd"):
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        use_news = st.checkbox("News features", value=True)
    with a2:
        use_options = st.checkbox("Options/IV features", value=True)
    with a3:
        regime_gate = st.checkbox("Regime-gating", value=True)
    with a4:
        build_portfolio = st.checkbox("Portefeuille optimaliseren", value=True)
    b1, b2, b3 = st.columns(3)
    with b1:
        max_w = st.slider("Max gewicht per naam", 0.05, 0.5, 0.25, step=0.05)
    with b2:
        risk_averse = st.slider("Risico-aversion (λ)", 1.0, 10.0, 5.0, step=0.5)
    with b3:
        turn_pen = st.slider("Turnover-penalty (γ)", 0.0, 0.05, 0.01, step=0.005)
    exec1, exec2, exec3 = st.columns(3)
    with exec1:
        max_part = st.slider("POV max participatie", 0.05, 0.5, 0.1, step=0.05)
    with exec2:
        max_spread_bps = st.slider("Max spread (bps) voor child-orders", 5, 50, 15, step=1)
    with exec3:
        default_qty = st.number_input("Doelwaarde per naam (stuks)", 1, 10000, 100)

run = st.button("Run Screener")

tabs = st.tabs(["📋 Kandidaatselectie", "📈 Portefeuille", "🔍 Uitleg", "⚙️ Uitvoering", "🧾 Export"])
tab_sel, tab_pf, tab_explain, tab_exec, tab_export = tabs

if run:
    with st.spinner("Data en modellen..."):
        prices = fetch_prices(tickers, lookback_days=int(lookback))

        # ====== Build base features & target per ticker ======
        rows = []
        for t, df in prices.items():
            if df.empty or "Close" not in df.columns:
                continue
            c = pd.to_numeric(df["Close"], errors="coerce").dropna()
            r = c.pct_change().fillna(0.0)
            m = pd.DataFrame(index=c.index)
            m["ret1"] = r.shift(1)
            m["ret5"] = c.pct_change(5).shift(1)
            m["vol20"] = r.rolling(20).std().shift(1)
            m["mom20"] = r.rolling(20).mean().shift(1)
            m["bbw"] = (pd.to_numeric(df["High"], errors="coerce") - pd.to_numeric(df["Low"], errors="coerce")).rolling(20).mean().shift(1) / (c.shift(1)+1e-9)
            y = c.pct_change(int(horizon)).shift(-int(horizon)).reindex(m.index)
            y_bin = (y > 0).astype(int)
            m["ticker"] = t
            m["target"] = y.values
            m["y_bin"] = y_bin.values
            rows.append(m.dropna())
        if not rows:
            st.error("Geen bruikbare data.")
            st.stop()
        feats = pd.concat(rows).reset_index().rename(columns={"index":"Date"})
        base_cols = ["ret1","ret5","vol20","mom20","bbw"]

        # ====== Base model + conformal ======
        X = feats[base_cols]
        y = feats["y_bin"].astype(int)
        n = len(X)
        cut = int(n*0.7)
        X_train, y_train = X.iloc[:cut], y.iloc[:cut]
        X_cal, y_cal = X.iloc[cut:], y.iloc[cut:]
        base_ctor = lambda: GradientBoostingClassifier(random_state=42)
        base_model = base_ctor(); base_model.fit(X_train, y_train)
        p_cal = base_model.predict_proba(X_cal)[:,1]
        tau = calibrate_tau_precision(y_cal.values, p_cal, eps=float(eps))

        # ====== Out-of-fold probs for meta-labeling ======
        oof_p = time_series_oof_probs(base_ctor, X, y, n_splits=5)
        feats["p_oof"] = oof_p
        # Directionele signalen uit OOF
        feats["dir_sig"] = np.where(feats["p_oof"]>0.5, 1.0, -1.0)

        # ====== Triple-Barrier labels per ticker (echte prijslabels) ======
        tb_labels_all = []
        for t in feats["ticker"].unique():
            sub = feats[feats["ticker"]==t].copy()
            # we hebben geen Close in feats; reconstrueer via cumulatieve returns is foutgevoelig
            # haal Close opnieuw voor juist label
            dfp = prices.get(t)
            if dfp is None or dfp.empty: 
                continue
            tb = triple_barrier_labels(dfp["Close"], tp=0.04, sl=0.03, max_hold=int(horizon))
            tb = tb.reindex(sub["Date"]).fillna(0.0)
            tb_labels_all.append(pd.DataFrame({"idx": sub.index, "tb": tb.values}))
        if tb_labels_all:
            tb_df = pd.concat(tb_labels_all).set_index("idx").sort_index()
            feats["tb"] = tb_df["tb"].reindex(feats.index).fillna(0.0)
        else:
            feats["tb"] = 0.0

        # ====== Meta labels en meta-model ======
        meta_y = meta_label_from_direction(pd.Series(feats["dir_sig"], index=feats.index), pd.Series(feats["tb"], index=feats.index))
        meta_X = feats[base_cols]  # je kunt news/options toevoegen voor nog meer kracht
        from sklearn.ensemble import RandomForestClassifier
        meta_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42).fit(meta_X, meta_y)

        # ====== Actuele features per ticker ======
        last_rows = []
        for t in feats["ticker"].unique():
            sub = feats[feats["ticker"]==t]
            if len(sub) < 120: 
                continue
            last_rows.append(sub.iloc[[-1]][["ticker","Date"]+base_cols+["target","y_bin"]])
        X_now = pd.concat(last_rows).set_index("ticker")

        # News & options & microstructure ======
        if use_news:
            items = get_news(tickers=list(X_now.index), limit_per=6, provider="auto")
            nf = build_news_features(items)
            X_now = X_now.join(nf.set_index("ticker"), on="ticker")
        else:
            X_now["claim_strength"] = 0.0
            X_now["share_of_voice"] = 0.0
        if use_options:
            opt_rows = [{"ticker": t, **options_features(t)} for t in X_now.index]
            of = pd.DataFrame(opt_rows).set_index("ticker")
            X_now = X_now.join(of, on="ticker")
        else:
            X_now["iv_skew"] = np.nan; X_now["iv_level"] = np.nan; X_now["oi_surge"] = np.nan
        mic_rows = [{"ticker": t, **intraday_features(t)} for t in X_now.index]
        mic = pd.DataFrame(mic_rows).set_index("ticker")
        X_now = X_now.join(mic, on="ticker")

        # ====== Regime gating ======
        pivot = feats.pivot_table(index="Date", columns="ticker", values="ret1").fillna(0.0)
        reg = regime_features(pivot.mean(axis=1))
        expert = pick_expert(reg)

        # ====== Selectie ======
        p_now = base_model.predict_proba(X_now[base_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0))[:,1]
        sel = p_now > tau
        meta_now = meta_model.predict_proba(X_now[base_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0))[:,1] > 0.5
        q_models = fit_quantiles(X, feats["target"].astype(float))
        q_pred = predict_quantiles(q_models, X_now[base_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0))
        q_dec = decision_from_quantiles(q_pred, ret_thresh=0.04)
        take = sel & meta_now & q_dec.values

        results = X_now.copy()
        results["p_hat"] = p_now
        results["meta_ok"] = meta_now
        results["q10"] = q_pred.get(0.1)
        results["q50"] = q_pred.get(0.5)
        results["q90"] = q_pred.get(0.9)
        results["take"] = take
        results["expert"] = expert

        # ====== Metrics header ======
        covg = coverage(p_now > tau)
        prec = precision_at_mask(y_cal.values, p_cal, tau)
        m1, m2, m3 = st.columns(3)
        m1.metric("τ (filter)", f"{tau:.3f}")
        m2.metric("Precision@τ (cal)", f"{prec:.2f}")
        m3.metric("Coverage@τ (nu)", f"{covg:.2f}")

        # ====== Tab: Selectie ======
        with tab_sel:
            st.subheader("Kandidaten")
            view = results[["Date","p_hat","meta_ok","q10","q50","q90","share_of_voice","claim_strength","iv_level","iv_skew","oi_surge","range_contraction","vwap_diff","spread_proxy","expert","take"]].copy()
            view = view.sort_values("p_hat", ascending=False)
            st.dataframe(view, use_container_width=True)
            st.caption("Tip: sorteer op p_hat of q50 voor focus.")

        # ====== Tab: Portefeuille ======
        with tab_pf:
            if build_portfolio and results["take"].any():
                exp_ret = results.loc[results["take"], "q50"].rename("exp_ret")
                cov = feats.pivot_table(index="Date", columns="ticker", values="target").dropna().tail(120).cov()
                w = optimize_weights(exp_ret, cov, prev_w=None, risk_aversion=float(risk_averse), turn_penalty=float(turn_pen), max_w=float(max_w))
                st.subheader("Voorgestelde wegingen")
                st.table(w.to_frame("weight"))
            else:
                st.info("Geen portefeuille gebouwd.")

        # ====== Tab: Uitleg ======
        with tab_explain:
            try:
                imp = permutation_importance(base_model, X_train, y_train, metric=average_precision_score, n_repeats=5)
                st.subheader("Belangrijkste drivers")
                st.write(imp.head(5))
            except Exception:
                st.write("—")

        # ====== Tab: Uitvoering ======
        with tab_exec:
            st.subheader("POV-plan per geselecteerde naam")
            take_list = [t for t in results.index if results.loc[t, "take"]]
            if not take_list:
                st.info("Nog geen namen geselecteerd.")
            else:
                pick = st.selectbox("Kies ticker", take_list)
                side = st.selectbox("Side", ["BUY","SELL"], index=0)
                qty = st.number_input("Stuks", 1, 1_000_000, int(default_qty))
                plan = generate_execution_plan(pick, side, qty, max_participation=float(max_part), max_spread_bps=int(max_spread_bps))
                st.dataframe(plan, use_container_width=True)
                st.download_button("Download uitvoeringsplan (CSV)", plan.to_csv(index=False).encode("utf-8"), f"execution_{pick}.csv", "text/csv")

        # ====== Tab: Export ======
        with tab_export:
            out = results.reset_index().rename(columns={"index":"ticker"})
            st.download_button("Download resultaten (CSV)", out.to_csv(index=False).encode("utf-8"), "screener_pro_results.csv", "text/csv")
