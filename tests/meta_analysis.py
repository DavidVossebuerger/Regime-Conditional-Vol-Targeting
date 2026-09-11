"""Cross-asset meta-analysis.

After running the pipeline on many assets (typically hundreds), aggregate the
per-asset results into a single significance verdict.

Reads all `summary.json` files under outputs/equities/{asset}/ and
outputs/multi_asset/{asset}/, computes:
  - Pooled hit rate (% of assets where RCVT beats B&H)
  - Mean Δ Sharpe ± standard error
  - Wilcoxon signed-rank test (paired Sharpe deltas across assets)
  - Cohen's d for paired samples
  - Binomial test for hit rate vs 50% null

Writes outputs/meta_analysis/summary.json + prints the master table.

Usage:
    python tests/meta_analysis.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Need scipy.stats for the tests
from scipy import stats


def collect_summaries(root: Path) -> pd.DataFrame:
    """Walk outputs/{equities,multi_asset}/**/summary.json and collect metrics."""
    rows = []
    for summary_path in sorted(root.glob("*/summary.json")):
        try:
            with open(summary_path) as f:
                d = json.load(f)
        except Exception:
            continue
        asset = summary_path.parent.name
        universe = summary_path.parent.parent.name  # equities / multi_asset
        bh = d.get("headline_bh", {})
        rcvt = d.get("headline_hmm", {})
        if not bh or not rcvt:
            continue
        bh_sharpe = bh.get("sharpe")
        rcvt_sharpe = rcvt.get("sharpe")
        if bh_sharpe is None or rcvt_sharpe is None:
            continue
        rows.append({
            "universe": universe,
            "asset": asset,
            "bh_sharpe": float(bh_sharpe),
            "rcvt_sharpe": float(rcvt_sharpe),
            "delta_sharpe": float(rcvt_sharpe - bh_sharpe),
            "bh_max_dd": float(bh.get("max_dd", float("nan"))),
            "rcvt_max_dd": float(rcvt.get("max_dd", float("nan"))),
            "p_hmm_beats_bh": float(d.get("p_hmm_beats_bh", float("nan"))),
            "n_windows": int(d.get("n_windows", 0)),
            "n_test_bars": int(d.get("n_test_bars", 0)),
            "covered_bars": int(d.get("covered_bars", 0)),
        })
    return pd.DataFrame(rows)


def meta_stats(df: pd.DataFrame, label: str) -> dict:
    """Compute meta-statistics for one universe."""
    # Drop rows with NaN Sharpe (some old Russell 2000 summaries have null)
    df = df.dropna(subset=["delta_sharpe", "bh_sharpe", "rcvt_sharpe"])
    df = df[np.isfinite(df["delta_sharpe"])]
    if len(df) < 5:
        return {"universe": label, "n": len(df), "skipped": True}
    deltas = df["delta_sharpe"].values
    bh = df["bh_sharpe"].values
    rcvt = df["rcvt_sharpe"].values
    n = len(deltas)
    hit_rate = float((deltas > 0).mean())
    mean_d = float(np.mean(deltas))
    se_d = float(np.std(deltas, ddof=1) / np.sqrt(n))
    # Wilcoxon signed-rank test (paired, non-parametric)
    try:
        w_stat, w_p = stats.wilcoxon(deltas, alternative="greater")
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    # Binomial test: P(at least k hits | null = 0.5)
    n_hits = int((deltas > 0).sum())
    b_p = float(stats.binomtest(n_hits, n, 0.5, alternative="greater").pvalue)
    # Cohen's d (paired)
    d_cohen = float(mean_d / np.std(deltas, ddof=1)) if np.std(deltas, ddof=1) > 0 else float("nan")
    return {
        "universe": label,
        "n": n,
        "hit_rate": hit_rate,
        "n_hits": n_hits,
        "mean_delta_sharpe": mean_d,
        "se_mean_delta_sharpe": se_d,
        "t_stat_paired": float(mean_d / se_d) if se_d > 0 else float("nan"),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p_one_sided": float(w_p),
        "binom_p_one_sided": b_p,
        "cohens_d_paired": d_cohen,
        "median_delta_sharpe": float(np.median(deltas)),
        "p10_delta_sharpe": float(np.percentile(deltas, 10)),
        "p90_delta_sharpe": float(np.percentile(deltas, 90)),
    }


def main():
    outputs = Path("outputs")
    rows = []
    for universe_dir in sorted(outputs.glob("*")):
        if not universe_dir.is_dir():
            continue
        df = collect_summaries(universe_dir)
        if df.empty:
            continue
        rows.append(df)
    if not rows:
        print(f"ERROR: no summary.json files found under outputs/")
        sys.exit(1)
    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(outputs / "meta_analysis" / "all_assets.csv", index=False,
                  errors="ignore") if False else None  # quiet
    out_dir = outputs / "meta_analysis"
    out_dir.mkdir(exist_ok=True)
    all_df.to_csv(out_dir / "all_assets.csv", index=False)

    # Per-universe meta stats
    print("\n=== Meta-analysis by universe ===\n")
    summary = []
    for universe, gdf in all_df.groupby("universe"):
        stats_d = meta_stats(gdf, universe)
        summary.append(stats_d)
        print(f"--- {universe} (n={stats_d['n']}) ---")
        print(f"  hit rate (RCVT > B&H):     {stats_d['hit_rate']*100:.1f}% ({stats_d['n_hits']}/{stats_d['n']})")
        print(f"  mean ΔSharpe:              {stats_d['mean_delta_sharpe']:+.3f} ± {stats_d['se_mean_delta_sharpe']:.3f} (SE)")
        print(f"  t-stat (paired):           {stats_d['t_stat_paired']:+.2f}")
        print(f"  Wilcoxon p (one-sided):    {stats_d['wilcoxon_p_one_sided']:.3g}")
        print(f"  binomial p (one-sided):    {stats_d['binom_p_one_sided']:.3g}")
        print(f"  Cohen's d (paired):        {stats_d['cohens_d_paired']:+.3f}")
        print(f"  median ΔSharpe:            {stats_d['median_delta_sharpe']:+.3f}")
        print(f"  p10/p90 ΔSharpe:           {stats_d['p10_delta_sharpe']:+.2f} / {stats_d['p90_delta_sharpe']:+.2f}")
        print()

    # Overall (all assets combined)
    overall = meta_stats(all_df, "ALL")
    print(f"--- ALL ({overall['n']} assets) ---")
    print(f"  hit rate (RCVT > B&H):     {overall['hit_rate']*100:.1f}% ({overall['n_hits']}/{overall['n']})")
    print(f"  mean ΔSharpe:              {overall['mean_delta_sharpe']:+.3f} ± {overall['se_mean_delta_sharpe']:.3f} (SE)")
    print(f"  t-stat (paired):           {overall['t_stat_paired']:+.2f}")
    print(f"  Wilcoxon p (one-sided):    {overall['wilcoxon_p_one_sided']:.3g}")
    print(f"  binomial p (one-sided):    {overall['binom_p_one_sided']:.3g}")
    print(f"  Cohen's d (paired):        {overall['cohens_d_paired']:+.3f}")
    print(f"  median ΔSharpe:            {overall['median_delta_sharpe']:+.3f}")
    print(f"  p10/p90 ΔSharpe:           {overall['p10_delta_sharpe']:+.2f} / {overall['p90_delta_sharpe']:+.2f}")

    # Save
    summary.append(overall)
    with open(out_dir / "meta_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWritten: {out_dir / 'meta_summary.json'}")
    print(f"Written: {out_dir / 'all_assets.csv'}")


if __name__ == "__main__":
    main()
