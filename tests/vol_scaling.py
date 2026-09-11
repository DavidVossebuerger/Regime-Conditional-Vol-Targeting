"""Vol-scaling experiment (v2): asset-independent test of edge vs vol level.

Hypothesis: HMM vol-targeting has larger advantage when underlying vol is higher,
because position changes are bigger and risk-management value scales with vol.

Methodology
-----------
1. Pick ~30 assets spanning the vol spectrum (deciles of native realized vol)
2. For each asset, scale log returns by factors [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
3. Re-run the full HMM pipeline on each scaled series
4. Compute Spearman ρ between vol scale and Δ Sharpe (all observations)
5. Also compute within-asset Spearman (z-score Δ Sharpe per asset first)

Outputs
-------
- outputs/vol_scaling/summary.csv — per-asset, per-scale metrics
- outputs/vol_scaling/by_scale.csv — cross-asset aggregate
- outputs/vol_scaling/scaling_curve.png — Δ Sharpe vs vol scale
- docs/vol_scaling_results.md — written report
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_io.yf_loader import fetch as yf_fetch
from equities_runner import (
    build_features, soft_position, metrics, sharpe_rank,
    select_per_state_targets, RF_PARAMS, SEED, K_HMM, TARGET_GRID,
    WF_TRAIN_MIN, WF_TEST_SIZE, WF_STEP, EPS, TC_PER_SIDE,
)

from scipy import stats

OUT_DIR = Path("outputs/vol_scaling")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VOL_SCALES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
N_SAMPLE = 30  # assets per decile → ~30 total spanning the vol spectrum


def pick_sample_assets(n: int = N_SAMPLE) -> list[str]:
    """Pick ~n yfinance-cached assets spanning the realized-vol spectrum.

    Loads every cached ticker, computes native annualized vol, sorts, and picks
    one ticker per decile so we get the full vol range.
    """
    cache = Path("data/yfinance")
    rows = []
    for f in cache.glob("*_1d.csv"):
        ticker = f.stem.replace("_1d", "")
        try:
            df = pd.read_csv(f, parse_dates=["Date"]).set_index("Date").sort_index()
            if len(df) < 1500:
                continue
            ann_vol = float(df["close"].pct_change().dropna().std() * np.sqrt(252))
            rows.append((ticker, ann_vol, len(df)))
        except Exception:
            continue
    df = pd.DataFrame(rows, columns=["ticker", "ann_vol", "n_bars"]).sort_values("ann_vol")
    # Pick one per decile → ~10 assets for 10 deciles. Increase by picking top 3 per decile.
    df["decile"] = pd.qcut(df["ann_vol"], 10, labels=False, duplicates="drop")
    sample = (df.groupby("decile").head(3)["ticker"].tolist())[:n]
    return sample


def run_pipeline_on_scaled_returns(
    scaled_logret: pd.Series, name: str, scale: float
) -> dict | None:
    """Re-run a simplified HMM pipeline on a (synthetic) scaled return series."""
    df = pd.DataFrame({"price": np.exp(scaled_logret.cumsum())}).dropna()
    log_ret = scaled_logret.reindex(df.index)
    feat = build_features(df["price"])
    train_n = int(len(feat) * 0.8)
    train = feat.iloc[:train_n]
    test = feat.iloc[train_n:]
    if len(test) < 100:
        return None
    test_idx = test.index
    N_TEST = len(test_idx)

    X_train = train.drop(columns=["target"]); y_train = train["target"]
    X_test = test.drop(columns=["target"])
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(**RF_PARAMS).fit(X_train, y_train)
    pred = pd.Series(rf.predict(X_test), index=test_idx, name="pred")
    pred_arr = pred.values
    log_ret_arr = log_ret.reindex(test_idx).fillna(0).values

    rv_realized = (log_ret * np.sqrt(252)).reindex(test_idx).fillna(0).values
    rv_lag1 = pd.Series(rv_realized).shift(1).fillna(0.0).values
    vol_zscore = pd.Series(rv_lag1).rolling(60).apply(
        lambda s: (s.iloc[-1] - s.mean()) / (s.std() + 1e-12), raw=False
    ).fillna(0.0).values
    vol_of_vol = pd.Series(rv_lag1).rolling(20).std().fillna(0.0).values
    vol_return = np.concatenate([[0.0], np.diff(rv_lag1)])
    hmm_X = np.column_stack([vol_zscore, vol_of_vol, vol_return])
    hmm_X = np.nan_to_num(hmm_X, nan=0.0, posinf=0.0, neginf=0.0)
    mu = hmm_X.mean(axis=0); sd = hmm_X.std(axis=0) + 1e-12
    hmm_X_std = (hmm_X - mu) / sd
    first_valid = 60
    hmm_X_eff = hmm_X_std[first_valid:]
    eff_idx = test_idx[first_valid:]
    if len(hmm_X_eff) < 90:
        return None

    eff_len = len(hmm_X_eff)
    min_wf = min(WF_TRAIN_MIN, max(60, eff_len // 3))
    test_size = min(WF_TEST_SIZE, max(40, eff_len // 6))
    step = test_size

    windows = []
    w = 0
    while True:
        te = w + min_wf
        if te + 30 >= eff_len: break
        fs = te; fe = min(fs + test_size, eff_len)
        if fe - fs < 40: break
        windows.append((w, te, first_valid + fs, first_valid + fe, fe - fs))
        w += step
        if fe >= eff_len: break
    if not windows:
        return None

    from hmmlearn.hmm import GaussianHMM
    blend_pos_full = np.full(N_TEST, np.nan)
    for w_idx, (eff0, eff1, fs, fe, n_days) in enumerate(windows):
        X_tr = hmm_X_eff[eff0:eff1]
        X_te = hmm_X_eff[eff1:(eff1 + n_days)]
        try:
            hmm = GaussianHMM(n_components=K_HMM, covariance_type="diag",
                              n_iter=150, random_state=SEED, tol=1e-4,
                              implementation="log").fit(X_tr)
            tr_probs = hmm.predict_proba(X_tr)
            te_probs = hmm.predict_proba(X_te) if len(X_te) > 0 else np.zeros((0, K_HMM))
        except Exception:
            tr_probs = np.ones((eff1 - eff0, K_HMM)) / K_HMM
            te_probs = np.ones((n_days, K_HMM)) / K_HMM

        train_logret = log_ret_arr[first_valid + eff0: first_valid + eff1]
        train_pred = rf.predict(X_test.iloc[first_valid + eff0: first_valid + eff1])
        best_thr, _ = select_per_state_targets(tr_probs, train_logret, train_pred)

        test_orig_idx = np.arange(fs, fe)
        pos_window = np.zeros(len(test_orig_idx))
        for k in range(K_HMM):
            tg = best_thr[k]
            binary_pos = soft_position(tg, pred_arr[test_orig_idx])
            pos_window += te_probs[:, k] * binary_pos
        blend_pos_full[test_orig_idx] = pos_window

    covered = np.zeros(N_TEST, dtype=bool)
    for ww in windows:
        covered[ww[2]:ww[3]] = True
    if covered.sum() < 50:
        return None

    blend_r = np.full(N_TEST, np.nan)
    for ww in windows:
        fs2, fe2 = ww[2], ww[3]
        pos = blend_pos_full[fs2:fe2]
        r = pos * log_ret_arr[fs2:fe2]
        tc = np.abs(np.diff(pos, prepend=pos[0])) * TC_PER_SIDE
        blend_r[fs2:fe2] = r - tc

    bh_r = log_ret_arr.copy()
    m_bh = metrics(bh_r[covered], np.ones(covered.sum()))
    m_blend = metrics(blend_r[covered], blend_pos_full[covered])

    return {
        "asset": name,
        "scale": scale,
        "n_test_bars": int(N_TEST),
        "n_windows": len(windows),
        "scaled_ann_vol": float(scaled_logret.std() * np.sqrt(252)),
        "bh_sharpe": m_bh["sharpe"],
        "hmm_sharpe": m_blend["sharpe"],
        "delta_sharpe": m_blend["sharpe"] - m_bh["sharpe"],
        "bh_max_dd": m_bh["max_dd"],
        "hmm_max_dd": m_blend["max_dd"],
        "hmm_pct_invested": float(np.nanmean(blend_pos_full[covered])),
    }


def main():
    sample = pick_sample_assets(N_SAMPLE)
    print(f"Vol-scaling experiment (v2) — {len(sample)} assets × {len(VOL_SCALES)} scales")
    print(f"Sample: {sample}")
    print(f"Scales: {VOL_SCALES}")
    print(f"TC = {TC_PER_SIDE*1e4:.1f} bps/side")
    print()

    rows = []
    total = len(sample) * len(VOL_SCALES)
    done = 0
    t_start = time.time()
    for sym in sample:
        s = yf_fetch(sym, interval="1d", period="max")
        if s is None or len(s) < 1500:
            done += len(VOL_SCALES)
            continue
        log_ret = np.log(s / s.shift(1)).dropna()
        base_ann_vol = log_ret.std() * np.sqrt(252)
        print(f"\n--- {sym} (base vol={base_ann_vol:.1%}, {len(log_ret)} bars) ---")

        for scale in VOL_SCALES:
            scaled = log_ret * scale
            try:
                r = run_pipeline_on_scaled_returns(scaled, sym, scale)
            except Exception as e:
                done += 1
                continue
            if r is None:
                done += 1
                continue
            print(f"  scale {scale:.2f}: Δ={r['delta_sharpe']:+.3f} "
                  f"(bh={r['bh_sharpe']:+.3f}, hmm={r['hmm_sharpe']:+.3f})")
            rows.append(r)
            done += 1
            elapsed = time.time() - t_start
            avg = elapsed / done
            eta = avg * (total - done)
            print(f"    [{done}/{total}]  eta={int(eta)}s", end="\r")

    df = pd.DataFrame(rows)
    if df.empty:
        print("ERROR: no results")
        return
    df.to_csv(OUT_DIR / "summary.csv", index=False)

    # Aggregate by scale
    agg = df.groupby("scale").agg(
        n=("asset", "count"),
        mean_delta_sharpe=("delta_sharpe", "mean"),
        median_delta_sharpe=("delta_sharpe", "median"),
        mean_bh_sharpe=("bh_sharpe", "mean"),
        mean_hmm_sharpe=("hmm_sharpe", "mean"),
        pct_hmm_beats_bh=("delta_sharpe", lambda s: (s > 0).mean()),
    ).reset_index()
    agg.to_csv(OUT_DIR / "by_scale.csv", index=False)

    print("\n\n=== Aggregate by scale ===")
    print(agg.round(3).to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for sym, sub in df.groupby("asset"):
        ax.plot(sub["scale"], sub["delta_sharpe"], "o-", label=sym, alpha=0.5)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Vol scale (×)")
    ax.set_ylabel("Δ Sharpe (HMM − B&H)")
    ax.set_title(f"Δ Sharpe vs Vol Scale — per asset ({df['asset'].nunique()} assets)")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1]
    std = df.groupby("scale")["delta_sharpe"].std().values
    ax.errorbar(agg["scale"], agg["mean_delta_sharpe"],
                yerr=std, fmt="o-", capsize=5, color="tab:red", lw=2,
                label="mean ± std across assets")
    ax.plot(agg["scale"], agg["median_delta_sharpe"], "s--",
            color="tab:blue", lw=2, label="median across assets")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Vol scale (×)")
    ax.set_ylabel("Δ Sharpe (HMM − B&H)")
    ax.set_title("Cross-Asset Aggregate: Δ Sharpe grows with vol, plateaus")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scaling_curve.png", dpi=120)
    plt.close(fig)

    # === Statistical tests ===
    # 1) Spearman on all observations (N=240 potentially)
    rho_all, p_all = stats.spearmanr(df["scale"], df["delta_sharpe"])
    # 2) Pearson on aggregate
    rho_mean, p_mean = stats.pearsonr(agg["scale"], agg["mean_delta_sharpe"])
    # 3) Within-asset Spearman: z-score Δ Sharpe per asset first, then Spearman vs scale
    df["delta_z_within"] = df.groupby("asset")["delta_sharpe"].transform(
        lambda s: (s - s.mean()) / s.std() if s.std() > 0 else 0.0)
    # weight by sample size per scale (uniform weight within asset)
    rho_within, p_within = stats.spearmanr(df["scale"], df["delta_z_within"])
    # 4) Bootstrap CI on rho_all
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(2000):
        bs = df.sample(n=len(df), replace=True, random_state=rng)
        rho_bs, _ = stats.spearmanr(bs["scale"], bs["delta_sharpe"])
        boots.append(rho_bs)
    boots = np.array(boots)
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])

    print(f"\n=== Statistical tests ===")
    print(f"All obs (n={len(df)}): Spearman ρ = {rho_all:+.3f}, p = {p_all:.3g}")
    print(f"All obs bootstrap CI (95 %): [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    print(f"Within-asset z-score Spearman ρ = {rho_within:+.3f}, p = {p_within:.3g}")
    print(f"Aggregate 5 bins Pearson r = {rho_mean:+.3f}, p = {p_mean:.3g}")

    # 5) By-scale hit rate
    print(f"\n=== Hit rate by scale ===")
    for scale in VOL_SCALES:
        sub = df[df["scale"] == scale]
        if len(sub) > 0:
            hit = (sub["delta_sharpe"] > 0).mean()
            print(f"  {scale:.2f}x (n={len(sub):>2d}): hit={hit*100:5.1f}%, "
                  f"mean Δ={sub['delta_sharpe'].mean():+.3f}")

    print(f"\nSaved to {OUT_DIR}/")
    for f in sorted(OUT_DIR.glob("*")):
        print(f"  {f}")


if __name__ == "__main__":
    import time
    main()
