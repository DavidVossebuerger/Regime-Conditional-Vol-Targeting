"""Deep diagnostic for XRP under-performance in the HMM wc-feat pipeline.

Does NOT modify multi_asset_risk.py. Imports its helpers and reuses them with
patched K_HMM / chosen_eta, so we can sweep configuration space.

Outputs:
  results_smoke/diag_xrp/
    k_sweep.csv
    eta_sweep.csv
    yearly_breakdown.csv
    vol_regime_decomposition.csv
    vov_comparison.csv
    xrp_eth_corr_experiment.csv
    figures/*.png
"""
from __future__ import annotations
from itertools import product
from pathlib import Path
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestRegressor

# import the existing pipeline as-is
import multi_asset_risk as mar

warnings.filterwarnings("ignore")

OUT = Path("results_smoke/diag_xrp")
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# ---- patch deterministic seed and reduce verbosity (no logic changes) ----
mar.SEED = 42

# ---- dataset ----
xrp = mar.load_asset("csv:xrp_usd_1h.csv")
eth = mar.load_asset("parquet:1m:crypto_ETH_USD")
btc = mar.load_asset("parquet:1m:crypto_BTC_USD")
print(f"XRP bars: {len(xrp)}  ETH bars: {0 if eth is None else len(eth)}  BTC bars: {0 if btc is None else len(btc)}")


# =========================================================================
# Core pipeline run with overridable K_HMM and fixed eta
# Mirrors multi_asset_risk.run_one_asset exactly so apples-to-apples.
# =========================================================================
def run_pipeline(close: pd.Series, K_HMM: int, eta: float, add_xrp_eth_corr: bool = False):
    log_ret_full = np.log(close / close.shift(1)).dropna()
    feat = mar.build_features(close)
    train_n = int(len(feat) * mar.SPLIT)
    train = feat.iloc[:train_n]
    test = feat.iloc[train_n:]
    test_idx = test.index
    N_TEST = len(test_idx)
    X_train = train.drop(columns=["target"]); y_train = train["target"]
    X_test = test.drop(columns=["target"])
    rf = mar.RandomForestRegressor(**mar.RF_PARAMS).fit(X_train, y_train)
    pred_arr = rf.predict(X_test)
    log_ret_arr = log_ret_full.reindex(test_idx).fillna(0).values

    rv_realized = (log_ret_full * np.sqrt(mar.PERIODS_PER_YEAR)).reindex(test_idx).fillna(0).values
    rv_lag1 = pd.Series(rv_realized).shift(1).fillna(0.0).values
    vol_zscore = pd.Series(rv_lag1).rolling(168).apply(
        lambda s: (s.iloc[-1] - s.mean()) / (s.std() + 1e-12), raw=False
    ).fillna(0.0).values
    vol_of_vol = pd.Series(rv_lag1).rolling(72).std().fillna(0.0).values
    vol_return = np.concatenate([[0.0], np.diff(rv_lag1)])
    # worst-case feature with chosen eta
    wc_full = mar.wc_series(log_ret_arr, eta, mar.WC_WINDOW)
    wc_test = pd.Series(wc_full, index=test_idx).shift(1).fillna(0.0).values

    cols = [vol_zscore, vol_of_vol, vol_return, wc_test]

    # optional extra: rolling 24h corr with ETH
    extra_label = None
    if add_xrp_eth_corr and eth is not None:
        eth_ret = np.log(eth / eth.shift(1)).reindex(test_idx).fillna(0)
        # 24h rolling correlation of hourly log returns
        xrp_ret_s = pd.Series(log_ret_arr, index=test_idx)
        roll_corr = xrp_ret_s.rolling(24).corr(eth_ret).fillna(0.0).values
        cols.append(roll_corr)
        extra_label = "xrp_eth_corr24"

    hmm_X = np.column_stack(cols)
    hmm_X = np.nan_to_num(hmm_X, nan=0.0, posinf=0.0, neginf=0.0)
    mu = hmm_X.mean(axis=0); sd = hmm_X.std(axis=0) + 1e-12
    hmm_X_std = ((hmm_X - mu) / sd)
    first_valid = 168
    hmm_X_eff = hmm_X_std[first_valid:]
    eff_idx = test_idx[first_valid:]

    # Walk-forward windows identical to original
    windows = []
    w = 0
    while True:
        te = w + mar.WF_TRAIN_MIN
        if te + 168 >= len(eff_idx): break
        fs_eff = te
        fe_eff = min(fs_eff + mar.WF_TEST_SIZE, len(eff_idx))
        if fe_eff - fs_eff < 720: break
        windows.append((w, te, first_valid + fs_eff, first_valid + fe_eff, fe_eff - fs_eff))
        w += mar.WF_STEP
        if fe_eff >= len(eff_idx): break

    blend_pos_full = np.full(N_TEST, np.nan)
    for w_idx, (eff0, eff1, fs, fe, n_days) in enumerate(windows):
        X_tr = hmm_X_eff[eff0:eff1]
        X_te = hmm_X_eff[eff1:(eff1 + n_days)]
        try:
            hmm = GaussianHMM(n_components=K_HMM, covariance_type="diag",
                              n_iter=150, random_state=mar.SEED, tol=1e-4,
                              implementation="log").fit(X_tr)
            tr_probs = hmm.predict_proba(X_tr)
            te_probs = hmm.predict_proba(X_te) if len(X_te) > 0 else np.zeros((0, K_HMM))
        except Exception:
            tr_probs = np.ones((eff1 - eff0, K_HMM)) / K_HMM
            te_probs = np.ones((n_days, K_HMM)) / K_HMM

        train_logret = log_ret_arr[first_valid + eff0: first_valid + eff1]
        train_pred = rf.predict(X_test.iloc[first_valid + eff0: first_valid + eff1])
        best_thr, _ = mar.select_per_state_targets(tr_probs, train_logret, train_pred)
        test_orig_idx = np.arange(fs, fe)
        pos_window = np.zeros(len(test_orig_idx))
        for k in range(K_HMM):
            tg = best_thr[k]
            binary_pos = mar.soft_position(tg, pred_arr[test_orig_idx])
            pos_window += te_probs[:, k] * binary_pos
        blend_pos_full[test_orig_idx] = pos_window

    covered = np.zeros(N_TEST, dtype=bool)
    for ww in windows:
        covered[ww[2]:ww[3]] = True

    blend_r = np.full(N_TEST, np.nan)
    for ww in windows:
        fs, fe = ww[2], ww[3]
        pos = blend_pos_full[fs:fe]
        r = pos * log_ret_arr[fs:fe]
        tc = np.abs(np.diff(pos, prepend=pos[0])) * mar.TC_PER_SIDE
        blend_r[fs:fe] = r - tc

    bh_r = log_ret_arr.copy()
    return dict(
        K=K_HMM, eta=eta, extra=extra_label,
        test_idx=test_idx, log_ret=log_ret_arr,
        bh_r=bh_r, blend_r=blend_r, blend_pos=blend_pos_full, covered=covered,
        windows=windows, rf=rf, X_test=X_test, pred_arr=pred_arr,
        first_valid=first_valid,
    )


# =========================================================================
# 1) Yearly breakdown using ORIGINAL config (K=3, eta=1.0)
# =========================================================================
print("\n[1] Original-config run for per-year analysis (K=3, eta=1.0)")
base = run_pipeline(xrp, K_HMM=3, eta=1.0)

test_idx = base["test_idx"]
bh_r = base["bh_r"]; blend_r = base["blend_r"]; blend_pos = base["blend_pos"]
covered = base["covered"]

rows = []
for yr in sorted(set(test_idx.year)):
    mask = (test_idx.year == yr) & covered
    if mask.sum() < 30: continue
    mb = mar.metrics(bh_r[mask], np.ones(mask.sum()))
    mh = mar.metrics(blend_r[mask], blend_pos[mask])
    rows.append({
        "year": yr, "n_hours": int(mask.sum()),
        "bh_sharpe": mb["sharpe"], "hmm_sharpe": mh["sharpe"],
        "sharpe_delta": mh["sharpe"] - mb["sharpe"],
        "bh_ann_ret": mb["ann_ret"], "hmm_ann_ret": mh["ann_ret"],
        "hmm_max_dd": mh["max_dd"], "hmm_pct_inv": float(blend_pos[mask].mean()),
        "hmm_ann_vol": mh["ann_vol"],
    })
yearly_df = pd.DataFrame(rows)
print(yearly_df.round(3).to_string(index=False))
yearly_df.to_csv(OUT / "yearly_breakdown.csv", index=False)


# =========================================================================
# 2) K_HMM sweep
# =========================================================================
print("\n[2] K_HMM sweep (eta=1.0)")
k_rows = []
k_runs = {}
for K in [2, 3, 4, 5]:
    r = run_pipeline(xrp, K_HMM=K, eta=1.0)
    k_runs[K] = r
    m_bh = mar.metrics(r["bh_r"][r["covered"]], np.ones(r["covered"].sum()))
    m_h = mar.metrics(r["blend_r"][r["covered"]], r["blend_pos"][r["covered"]])
    k_rows.append({
        "K": K,
        "bh_sharpe": m_bh["sharpe"], "hmm_sharpe": m_h["sharpe"],
        "sharpe_delta": m_h["sharpe"] - m_bh["sharpe"],
        "bh_max_dd": m_bh["max_dd"], "hmm_max_dd": m_h["max_dd"],
        "hmm_ann_vol": m_h["ann_vol"], "hmm_pct_inv": float(r["blend_pos"][r["covered"]].mean()),
        "pct_within_target": m_h["pct_within"],
    })
k_df = pd.DataFrame(k_rows)
print(k_df.round(3).to_string(index=False))
k_df.to_csv(OUT / "k_sweep.csv", index=False)


# =========================================================================
# 3) Eta sensitivity
# =========================================================================
print("\n[3] Eta sensitivity (K=3)")
e_rows = []
for eta in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
    r = run_pipeline(xrp, K_HMM=3, eta=eta)
    m_bh = mar.metrics(r["bh_r"][r["covered"]], np.ones(r["covered"].sum()))
    m_h = mar.metrics(r["blend_r"][r["covered"]], r["blend_pos"][r["covered"]])
    e_rows.append({
        "eta": eta,
        "bh_sharpe": m_bh["sharpe"], "hmm_sharpe": m_h["sharpe"],
        "sharpe_delta": m_h["sharpe"] - m_bh["sharpe"],
        "bh_max_dd": m_bh["max_dd"], "hmm_max_dd": m_h["max_dd"],
        "hmm_ann_vol": m_h["ann_vol"], "hmm_pct_inv": float(r["blend_pos"][r["covered"]].mean()),
        "pct_within_target": m_h["pct_within"],
    })
e_df = pd.DataFrame(e_rows)
print(e_df.round(3).to_string(index=False))
e_df.to_csv(OUT / "eta_sweep.csv", index=False)


# =========================================================================
# 4) Vol-regime decomposition (high-vol vs low-vol hours) — base run
# =========================================================================
print("\n[4] Vol-regime decomposition")
# Use rolling 24h ann vol from base run, threshold = 0.6 (target)
log_ret_s = pd.Series(base["log_ret"], index=test_idx)
roll_vol = log_ret_s.rolling(24).std(ddof=1) * np.sqrt(mar.PERIODS_PER_YEAR)
roll_vol = roll_vol.fillna(0)
hi = (roll_vol > 0.6).values & covered
lo = (roll_vol <= 0.6).values & covered
vol_rows = []
for label, mask in [("high_vol", hi), ("low_vol", lo), ("all_covered", covered)]:
    if mask.sum() < 30: continue
    mb = mar.metrics(base["bh_r"][mask], np.ones(mask.sum()))
    mh = mar.metrics(base["blend_r"][mask], base["blend_pos"][mask])
    vol_rows.append({
        "regime": label, "n_hours": int(mask.sum()),
        "bh_sharpe": mb["sharpe"], "hmm_sharpe": mh["sharpe"],
        "sharpe_delta": mh["sharpe"] - mb["sharpe"],
        "bh_ann_ret": mb["ann_ret"], "hmm_ann_ret": mh["ann_ret"],
        "hmm_max_dd": mh["max_dd"], "hmm_pct_inv": float(base["blend_pos"][mask].mean()),
    })
v_df = pd.DataFrame(vol_rows)
print(v_df.round(3).to_string(index=False))
v_df.to_csv(OUT / "vol_regime_decomposition.csv", index=False)


# =========================================================================
# 5) Volatility-of-volatility comparison: BTC vs ETH vs XRP
# =========================================================================
print("\n[5] VoV comparison")
def vov_stats(close, name):
    r = np.log(close / close.shift(1)).dropna()
    rv = r * np.sqrt(mar.PERIODS_PER_YEAR)
    # rolling 72h vol (matches pipeline's vol_of_vol feature)
    rv_lag1 = rv.shift(1).fillna(0.0)
    vov = pd.Series(rv_lag1).rolling(72).std().dropna()
    return {
        "asset": name,
        "n_obs": int(len(r)),
        "ann_vol_median": float(rv.median()),
        "ann_vol_p95": float(rv.quantile(0.95)),
        "vol_of_vol_median": float(vov.median()),
        "vol_of_vol_p95": float(vov.quantile(0.95)),
        "vol_of_vol_mean": float(vov.mean()),
        "kurtosis_logret": float(r.kurtosis()),
    }

vov_rows = [vov_stats(xrp, "XRP")]
if eth is not None: vov_rows.append(vov_stats(eth, "ETH"))
if btc is not None: vov_rows.append(vov_stats(btc, "BTC"))
vov_df = pd.DataFrame(vov_rows)
print(vov_df.round(4).to_string(index=False))
vov_df.to_csv(OUT / "vov_comparison.csv", index=False)


# =========================================================================
# 6) Cross-asset correlation with ETH (added as HMM feature)
# =========================================================================
print("\n[6] XRP-ETH rolling correlation as HMM feature")
# baseline (K=3, eta=1.0)
base2 = run_pipeline(xrp, K_HMM=3, eta=1.0, add_xrp_eth_corr=False)
# augmented
aug = None
if eth is not None:
    aug = run_pipeline(xrp, K_HMM=3, eta=1.0, add_xrp_eth_corr=True)
def headline(r):
    m_bh = mar.metrics(r["bh_r"][r["covered"]], np.ones(r["covered"].sum()))
    m_h = mar.metrics(r["blend_r"][r["covered"]], r["blend_pos"][r["covered"]])
    return m_bh, m_h

mb1, mh1 = headline(base2)
mb2, mh2 = headline(aug) if aug is not None else (mb1, mh1)
corr_rows = [
    {"config": "baseline_K3_eta1", "hmm_sharpe": mh1["sharpe"], "hmm_max_dd": mh1["max_dd"],
     "hmm_ann_vol": mh1["ann_vol"], "hmm_pct_inv": float(base2["blend_pos"][base2["covered"]].mean())},
]
if aug is not None:
    corr_rows.append({"config": "plus_xrp_eth_corr24", "hmm_sharpe": mh2["sharpe"],
                      "hmm_max_dd": mh2["max_dd"], "hmm_ann_vol": mh2["ann_vol"],
                      "hmm_pct_inv": float(aug["blend_pos"][aug["covered"]].mean())})
corr_df = pd.DataFrame(corr_rows)
print(corr_df.round(3).to_string(index=False))
corr_df.to_csv(OUT / "xrp_eth_corr_experiment.csv", index=False)


# =========================================================================
# 6b) Combined best-case: K=4 + eta=4 + ETH corr feature
# =========================================================================
print("\n[6b] Combined best-case run (K=4, eta=4, +xrp_eth_corr)")
combined = None
if eth is not None:
    combined = run_pipeline(xrp, K_HMM=4, eta=4.0, add_xrp_eth_corr=True)
    mc_bh = mar.metrics(combined["bh_r"][combined["covered"]], np.ones(combined["covered"].sum()))
    mc_h = mar.metrics(combined["blend_r"][combined["covered"]], combined["blend_pos"][combined["covered"]])
    print(f"  Combined K=4,eta=4,corr: HMM Sharpe={mc_h['sharpe']:.3f}  MaxDD={mc_h['max_dd']:.3f}  "
          f"vs B&H Sharpe={mc_bh['sharpe']:.3f} MaxDD={mc_bh['max_dd']:.3f}")
    combined_summary = {
        "config": "K4_eta4_ethcorr",
        "hmm_sharpe": mc_h["sharpe"], "hmm_max_dd": mc_h["max_dd"],
        "hmm_ann_vol": mc_h["ann_vol"], "hmm_pct_inv": float(combined["blend_pos"][combined["covered"]].mean()),
        "bh_sharpe": mc_bh["sharpe"], "bh_max_dd": mc_bh["max_dd"],
        "sharpe_delta": mc_h["sharpe"] - mc_bh["sharpe"],
    }
    (OUT / "combined_best.json").write_text(json.dumps(combined_summary, indent=2, default=str))


# =========================================================================
# 7) Position timeline visualization
# =========================================================================
print("\n[7] Building position-timeline plots")
fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True,
                          gridspec_kw={"height_ratios": [2, 2, 2, 2]})

# price
ax = axes[0]
ax.plot(test_idx, xrp.reindex(test_idx), lw=0.8, color="black")
ax.set_ylabel("XRP close (USD)"); ax.set_title("XRP test fold: price, vol, regime-positions, equity")
ax.grid(alpha=0.3)

# rolling vol
ax = axes[1]
ax.plot(test_idx, roll_vol, lw=0.7, color="tab:purple", label="rolling 24h ann.vol")
ax.axhline(0.6, color="k", lw=1, ls="--", label="vol target 60%")
ax.set_ylabel("ann. vol"); ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)

# HMM positions — color by state? Use blend_pos heatmap
ax = axes[2]
pos_plot = pd.Series(base["blend_pos"], index=test_idx).fillna(0)
ax.fill_between(test_idx, 0, pos_plot, color="tab:red", alpha=0.7, label="HMM wc-feat position")
ax.axhline(1.0, color="k", lw=0.5, ls=":", label="B&H = 1.0")
ax.set_ylabel("position"); ax.set_ylim(0, 1.2)
ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)

# equity
ax = axes[3]
for label, r, c in [("B&H", base["bh_r"], "tab:blue"), ("HMM wc-feat", base["blend_r"], "tab:red")]:
    cs = np.where(np.isnan(r), 0.0, np.where(np.isfinite(r), r, 0.0))
    eq = np.exp(np.cumsum(cs))
    eq = np.where(np.isnan(r), np.nan, eq)
    ax.plot(test_idx, eq, lw=0.9, color=c, label=label)
ax.set_ylabel("Equity (start=1)"); ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "position_timeline.png", dpi=110)
plt.close(fig)


# =========================================================================
# 8) Drawdown overlap plot: HMM "invested" vs drawdown
# =========================================================================
print("\n[8] Drawdown-overlap plot")
fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
# top: drawdowns
cs_bh = np.cumsum(pd.Series(base["bh_r"]).fillna(0))
peak_bh = np.maximum.accumulate(cs_bh); dd_bh = cs_bh - peak_bh
cs_h = np.cumsum(pd.Series(base["blend_r"]).fillna(0))
peak_h = np.maximum.accumulate(cs_h); dd_h = cs_h - peak_h
ax[0].fill_between(test_idx, dd_bh, 0, alpha=0.4, color="tab:blue", label="B&H DD")
ax[0].fill_between(test_idx, dd_h, 0, alpha=0.4, color="tab:red", label="HMM DD")
ax[0].set_ylabel("Drawdown (log)"); ax[0].legend(loc="lower left", fontsize=9); ax[0].grid(alpha=0.3)
ax[0].set_title("Drawdowns vs HMM investment stance")

# bottom: position with DD overlay
pos_plot = pd.Series(base["blend_pos"], index=test_idx).fillna(0)
ax[1].fill_between(test_idx, 0, pos_plot, color="tab:red", alpha=0.6, label="HMM position")
# shade periods where HMM is invested AND in drawdown
in_dd = (pd.Series(dd_h, index=test_idx) < -0.05).values
ax[1].fill_between(test_idx, 0, 1.2, where=in_dd, color="grey", alpha=0.15, label="HMM in DD<-5%")
ax[1].set_ylabel("position"); ax[1].set_ylim(0, 1.2); ax[1].legend(loc="upper left", fontsize=9); ax[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "dd_overlap.png", dpi=110)
plt.close(fig)


# =========================================================================
# 9) Per-window breakdown for K sweep (heat map)
# =========================================================================
print("\n[9] K-sweep per-window heatmap")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
wins_labels = [f"{base['test_idx'][w[2]].date()}\n→{base['test_idx'][w[3] - 1].date()}" for w in base["windows"]]
mat = np.zeros((len([2,3,4,5]), len(wins_labels)))
mat_delta = np.zeros_like(mat)
for i, K in enumerate([2, 3, 4, 5]):
    r = k_runs[K]
    for j, ww in enumerate(r["windows"]):
        fs, fe = ww[2], ww[3]
        m_h = mar.metrics(r["blend_r"][fs:fe], r["blend_pos"][fs:fe])
        m_b = mar.metrics(r["bh_r"][fs:fe], np.ones(fe - fs))
        mat[i, j] = m_h["sharpe"] if m_h["sharpe"] == m_h["sharpe"] else 0.0
        mat_delta[i, j] = (m_h["sharpe"] - m_b["sharpe"]) if m_h["sharpe"] == m_h["sharpe"] else 0.0

im = axes[0].imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-3, vmax=3)
axes[0].set_xticks(range(len(wins_labels))); axes[0].set_xticklabels(wins_labels, rotation=0, fontsize=7)
axes[0].set_yticks(range(4)); axes[0].set_yticklabels([f"K={k}" for k in [2,3,4,5]])
axes[0].set_title("HMM Sharpe per (K, window) — base config")
fig.colorbar(im, ax=axes[0])

im2 = axes[1].imshow(mat_delta, cmap="RdYlGn", aspect="auto", vmin=-3, vmax=3)
axes[1].set_xticks(range(len(wins_labels))); axes[1].set_xticklabels(wins_labels, rotation=0, fontsize=7)
axes[1].set_yticks(range(4)); axes[1].set_yticklabels([f"K={k}" for k in [2,3,4,5]])
axes[1].set_title("HMM-B&H Sharpe delta per (K, window)")
fig.colorbar(im2, ax=axes[1])
fig.tight_layout()
fig.savefig(FIG / "k_window_heatmap.png", dpi=110)
plt.close(fig)


# =========================================================================
# 10) VoV bar plot
# =========================================================================
print("\n[10] VoV bar plot")
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
xs = np.arange(len(vov_df))
ax[0].bar(xs, vov_df["vol_of_vol_median"], color=["tab:red", "tab:orange", "tab:blue"][: len(vov_df)])
ax[0].set_xticks(xs); ax[0].set_xticklabels(vov_df["asset"])
ax[0].set_ylabel("VoV (median)"); ax[0].set_title("Vol-of-Vol median")
ax[0].grid(alpha=0.3, axis="y")
ax[1].bar(xs, vov_df["kurtosis_logret"], color=["tab:red", "tab:orange", "tab:blue"][: len(vov_df)])
ax[1].set_xticks(xs); ax[1].set_xticklabels(vov_df["asset"])
ax[1].set_ylabel("Excess kurtosis of hourly log-returns"); ax[1].set_title("Tail heaviness")
ax[1].grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(FIG / "vov_comparison.png", dpi=110)
plt.close(fig)


# =========================================================================
# 11) Per-state-target distribution (does K=3 collapse? HMM states)
# =========================================================================
print("\n[11] Per-state diagnostics")
# Inspect learned HMM on a single window for K=3 vs K=5
state_rows = []
for K in [2, 3, 4, 5]:
    # re-fit on the largest training chunk
    feat = mar.build_features(xrp)
    train_n = int(len(feat) * mar.SPLIT)
    train = feat.iloc[:train_n]
    test = feat.iloc[train_n:]
    log_ret_full = np.log(xrp / xrp.shift(1)).dropna()
    test_idx = test.index
    log_ret_arr = log_ret_full.reindex(test_idx).fillna(0).values
    rv_realized = (log_ret_full * np.sqrt(mar.PERIODS_PER_YEAR)).reindex(test_idx).fillna(0).values
    rv_lag1 = pd.Series(rv_realized).shift(1).fillna(0.0).values
    vol_zscore = pd.Series(rv_lag1).rolling(168).apply(
        lambda s: (s.iloc[-1] - s.mean()) / (s.std() + 1e-12), raw=False
    ).fillna(0.0).values
    vol_of_vol = pd.Series(rv_lag1).rolling(72).std().fillna(0.0).values
    vol_return = np.concatenate([[0.0], np.diff(rv_lag1)])
    wc_full = mar.wc_series(log_ret_arr, 1.0, mar.WC_WINDOW)
    wc_test = pd.Series(wc_full, index=test_idx).shift(1).fillna(0.0).values
    hmm_X = np.column_stack([vol_zscore, vol_of_vol, vol_return, wc_test])
    hmm_X = np.nan_to_num(hmm_X, nan=0.0, posinf=0.0, neginf=0.0)
    mu = hmm_X.mean(axis=0); sd = hmm_X.std(axis=0) + 1e-12
    hmm_X_std = (hmm_X - mu) / sd
    # use second window (Oct 2025 - Jan 2026 = the painful one)
    eff0 = mar.WF_TRAIN_MIN
    eff1 = eff0 + mar.WF_TEST_SIZE
    try:
        hmm = GaussianHMM(n_components=K, covariance_type="diag", n_iter=200,
                          random_state=mar.SEED, tol=1e-4, implementation="log").fit(hmm_X_std[168 + eff0: 168 + eff1])
        means = hmm.means_
        # order states by vol_of_vol mean (feature index 1) so smaller=calmer
        order = np.argsort(means[:, 1])
        for new_k, old_k in enumerate(order):
            state_rows.append({
                "K": K, "state_rank": new_k,
                "vol_zscore_mean": float(means[old_k, 0]),
                "vol_of_vol_mean": float(means[old_k, 1]),
                "vol_return_mean": float(means[old_k, 2]),
                "wc_mean": float(means[old_k, 3]),
            })
    except Exception as e:
        state_rows.append({"K": K, "state_rank": -1, "error": str(e)})

state_df = pd.DataFrame(state_rows)
print(state_df.round(3).to_string(index=False))
state_df.to_csv(OUT / "state_means.csv", index=False)


# =========================================================================
# 11b) Equity-curve comparison across configs
# =========================================================================
print("\n[11b] Equity comparison plot")
configs = [
    ("B&H", base["bh_r"], "tab:blue"),
    ("default K=3 eta=1.0", base["blend_r"], "tab:red"),
    ("K=4 eta=1.0", k_runs[4]["blend_r"], "tab:green"),
    ("K=3 eta=4.0", None, "tab:orange"),  # filled below
]
# add eta=4 run
eta4_run = run_pipeline(xrp, K_HMM=3, eta=4.0)
configs[3] = ("K=3 eta=4.0", eta4_run["blend_r"], "tab:orange")
if combined is not None:
    configs.append(("K=4 eta=4.0 +ETH corr", combined["blend_r"], "tab:purple"))

fig, ax = plt.subplots(1, 1, figsize=(13, 6))
for label, r, c in configs:
    cs = np.where(np.isnan(r), 0.0, np.where(np.isfinite(r), r, 0.0))
    eq = np.exp(np.cumsum(cs))
    eq = np.where(np.isnan(r), np.nan, eq)
    ax.plot(test_idx, eq, lw=0.9, color=c, label=label)
ax.axhline(1.0, color="k", lw=0.5, ls=":")
ax.set_ylabel("Equity (start=1)"); ax.set_title("XRP test fold: config comparison")
ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "equity_comparison.png", dpi=110)
plt.close(fig)


# =========================================================================
# 12) Final summary json
# =========================================================================
summary = {
    "xrp_bars": int(len(xrp)),
    "test_period": f"{pd.Timestamp(test_idx[0]).date()} → {pd.Timestamp(test_idx[-1]).date()}",
    "k_sweep": k_df.to_dict(orient="records"),
    "eta_sweep": e_df.to_dict(orient="records"),
    "yearly_breakdown": yearly_df.to_dict(orient="records"),
    "vol_regime_decomposition": v_df.to_dict(orient="records"),
    "vov_comparison": vov_df.to_dict(orient="records"),
    "xrp_eth_corr_experiment": corr_df.to_dict(orient="records"),
}
def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o) if o == o else None
    if isinstance(o, (np.ndarray,)): return o.tolist()
    if isinstance(o, pd.Timestamp): return o.isoformat()
    return str(o)

(OUT / "diag_summary.json").write_text(json.dumps(summary, indent=2, default=conv))

print("\n=== Diagnostic finished ===")
print(f"Outputs in: {OUT}")
for f in sorted(OUT.rglob("*")):
    print(" -", f, f"({f.stat().st_size} bytes)" if f.is_file() else "(dir)")
