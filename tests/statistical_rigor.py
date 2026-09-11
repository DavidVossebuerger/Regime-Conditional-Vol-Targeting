"""Statistical significance tests for the HMM vol-targeting pipeline.

Reads existing summary CSVs and runs proper inference tests on the cross-asset
results. Does NOT re-fit any model — purely statistical analysis.

Tests performed
================
1. **One-sample t-test** on mean Δ Sharpe vs 0
   - Treat each asset as an independent observation
   - H0: mean Δ Sharpe = 0 (HMM has no edge over B&H)
   - Report t-statistic, df, p-value (two-sided), 95% CI, Cohen's d

2. **One-sample Wilcoxon signed-rank test** as non-parametric robustness
   - Less sensitive to outliers (e.g., OSCR with -1.94)

3. **Binomial test** on hit-rate
   - H0: P(HMM > B&H) = 0.5 (random chance)
   - Use exact binomial test for small N (crypto) and normal approximation for large N (Russell)

4. **Bootstrap CI on mean Δ Sharpe**
   - 10000 resamples of the cross-asset Δ Sharpe distribution
   - Report 95% CI

5. **Power analysis**
   - Effect size (Cohen's d) from observed mean/std
   - Required sample size for 80% power at α=0.05

Outputs
=======
- stdout: human-readable summary
- outputs/stat_rigor/results.json: machine-readable
- docs/statistical_rigor.md: human-readable report

Usage
=====
    python tests/statistical_rigor.py

Reads:
    outputs/multi_asset/per_asset_summary.csv
    outputs/russell2000_top200/summary.csv
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = Path("outputs/stat_rigor")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_deltas(csv_path: Path) -> pd.DataFrame | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "bh_sharpe" not in df.columns or "hmm_sharpe" not in df.columns:
        return None
    asset_col = "symbol" if "symbol" in df.columns else "asset"
    df["delta_sharpe"] = df["hmm_sharpe"] - df["bh_sharpe"]
    df["dd_cut"] = df["bh_max_dd"] - df["hmm_max_dd"]
    df = df.dropna(subset=["delta_sharpe"]).reset_index(drop=True)
    return df[[asset_col, "bh_sharpe", "hmm_sharpe", "delta_sharpe", "dd_cut"]] \
        .rename(columns={asset_col: "asset"})


def analyse(name: str, df: pd.DataFrame) -> dict:
    """Run the full battery of significance tests on one universe."""
    deltas = df["delta_sharpe"].values
    n = len(deltas)
    mean = float(np.mean(deltas))
    sd = float(np.std(deltas, ddof=1))
    se = sd / np.sqrt(n)
    n_pos = int(np.sum(deltas > 0))
    n_neg = int(np.sum(deltas < 0))

    # 1. One-sample t-test (H0: mean = 0)
    t_stat, t_p = stats.ttest_1samp(deltas, 0.0)
    t_ci = stats.t.interval(0.95, n - 1, loc=mean, scale=se)

    # 2. Wilcoxon signed-rank (non-parametric)
    try:
        w_stat, w_p = stats.wilcoxon(deltas, alternative="two-sided")
    except ValueError:
        # All zeros or other edge case
        w_stat, w_p = float("nan"), float("nan")

    # 3. Effect size (Cohen's d for paired / one-sample)
    cohens_d = mean / sd if sd > 0 else float("nan")

    # 4. Binomial test: P(k or more successes out of n | p=0.5)
    binom_p_two_sided = stats.binomtest(n_pos, n, 0.5, alternative="two-sided").pvalue
    binom_p_one_sided = stats.binomtest(n_pos, n, 0.5, alternative="greater").pvalue

    # 5. Bootstrap CI on mean
    rng = np.random.default_rng(42)
    boot_means = np.array([np.mean(rng.choice(deltas, size=n, replace=True)) for _ in range(10000)])
    boot_ci = (float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975)))

    # 6. Power analysis: required n for 80% power given current effect size
    # Use normal-approximation: n = ((z_α/2 + z_β) / d)² for two-sided
    if abs(cohens_d) > 1e-6:
        from scipy.stats import norm
        z_alpha = norm.ppf(0.975)
        z_beta = norm.ppf(0.80)
        required_n = ((z_alpha + z_beta) / abs(cohens_d)) ** 2
        required_n = int(np.ceil(required_n))
    else:
        required_n = float("nan")

    return {
        "name": name,
        "n_assets": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": int(np.sum(deltas == 0)),
        "hit_rate": n_pos / n,
        "mean_delta_sharpe": mean,
        "std_delta_sharpe": sd,
        "se_mean": se,
        "t_statistic": float(t_stat),
        "t_p_value": float(t_p),
        "t_95ci_low": float(t_ci[0]),
        "t_95ci_high": float(t_ci[1]),
        "wilcoxon_statistic": float(w_stat) if not np.isnan(w_stat) else None,
        "wilcoxon_p_value": float(w_p) if not np.isnan(w_p) else None,
        "cohens_d": float(cohens_d),
        "binom_p_two_sided": float(binom_p_two_sided),
        "binom_p_one_sided_greater": float(binom_p_one_sided),
        "bootstrap_95ci_low": boot_ci[0],
        "bootstrap_95ci_high": boot_ci[1],
        "required_n_for_80pct_power": required_n,
    }


def main():
    crypto_path = Path("outputs/multi_asset/per_asset_summary.csv")
    russell_path = Path("outputs/russell2000_top200/summary.csv")

    crypto_df = load_deltas(crypto_path)
    russell_df = load_deltas(russell_path)

    results = {}
    if crypto_df is not None:
        results["crypto"] = analyse("crypto_multi_asset", crypto_df)
    if russell_df is not None:
        results["russell2000_top200"] = analyse("russell2000_top200", russell_df)

    if not results:
        print("ERROR: no summary CSVs found. Run the pipelines first.")
        return

    # Print human-readable summary
    print("=" * 72)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 72)
    for name, r in results.items():
        print(f"\n--- {name.upper()} ({r['n_assets']} assets) ---")
        print(f"  Δ Sharpe:  mean = {r['mean_delta_sharpe']:+.4f}  std = {r['std_delta_sharpe']:.4f}  se = {r['se_mean']:.4f}")
        print(f"  Hit-rate:  {r['n_pos']}/{r['n_assets']} = {r['hit_rate']*100:.1f}% beat B&H")
        print(f"\n  1-sample t-test (H0: μ = 0)")
        print(f"     t = {r['t_statistic']:+.3f},  p = {r['t_p_value']:.4g},  95% CI = [{r['t_95ci_low']:+.4f}, {r['t_95ci_high']:+.4f}]")
        print(f"\n  Wilcoxon signed-rank (H0: median = 0)")
        if r['wilcoxon_p_value'] is not None:
            print(f"     W = {r['wilcoxon_statistic']:.1f},  p = {r['wilcoxon_p_value']:.4g}")
        print(f"\n  Effect size (Cohen's d): {r['cohens_d']:+.3f}")
        print(f"\n  Binomial test (H0: P(HMM>B&H) = 0.5)")
        print(f"     two-sided p = {r['binom_p_two_sided']:.4g}")
        print(f"     one-sided p (greater) = {r['binom_p_one_sided_greater']:.4g}")
        print(f"\n  Bootstrap CI on mean (10k resamples): [{r['bootstrap_95ci_low']:+.4f}, {r['bootstrap_95ci_high']:+.4f}]")
        print(f"\n  Required n for 80% power @ α=0.05: {r['required_n_for_80pct_power']} (have {r['n_assets']})")

    # Save JSON
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\n\nSaved to {OUT_DIR}/results.json")

    # Honest verdict
    print("\n" + "=" * 72)
    print("HONEST VERDICT")
    print("=" * 72)
    for name, r in results.items():
        p_t = r['t_p_value']
        p_binom = r['binom_p_two_sided']
        n = r['n_assets']
        verdict = "STRONG" if p_t < 0.001 and p_binom < 0.001 else \
                  "MODERATE" if p_t < 0.01 and p_binom < 0.01 else \
                  "MARGINAL" if p_t < 0.05 and p_binom < 0.05 else \
                  "INSUFFICIENT"
        cohens = r['cohens_d']
        cohens_label = "small" if abs(cohens) < 0.5 else \
                       "medium" if abs(cohens) < 0.8 else \
                       "large"
        print(f"  {name}: {verdict} (t p={p_t:.4g}, binomial p={p_binom:.4g}, Cohen's d={cohens:+.2f} = {cohens_label})")


if __name__ == "__main__":
    main()
