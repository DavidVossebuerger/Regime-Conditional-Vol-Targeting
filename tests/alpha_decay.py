"""Alpha decay test: does RCVT's edge fade over time?

For each asset's walk-forward windows, compute:
  - HMM test Sharpe (from per_window.csv)
  - BH test Sharpe (computed from yfinance cache over the same dates)
  - Δ Sharpe = HMM - BH

Then aggregate Δ Sharpe across assets by test_start year, and test whether
Δ Sharpe trends downward over time.

Methods:
  - OLS regression: Δ Sharpe ~ year
  - Spearman correlation: Δ Sharpe vs year
  - Mann-Kendall trend test (non-parametric)
"""
from __future__ import annotations
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_io.yf_loader import fetch as yf_fetch

OUT_DIR = Path("outputs/alpha_decay")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANCHORED_DIR = Path("outputs/equities_yf_anchored")


def compute_bh_sharpe(ticker: str, start, end, periods_per_year: int = 252) -> float | None:
    """Buy-and-hold Sharpe for a ticker over [start, end]."""
    s = yf_fetch(ticker, interval="1d", period="max", cache=True)
    if s is None:
        return None
    if isinstance(start, str):
        start = pd.Timestamp(start)
    if isinstance(end, str):
        end = pd.Timestamp(end)
    s = s.loc[start:end]
    if len(s) < 20:
        return None
    log_ret = np.log(s / s.shift(1)).dropna()
    sd = log_ret.std(ddof=1)
    if sd < 1e-10:
        return float("nan")
    return float(log_ret.mean() / sd * np.sqrt(periods_per_year))


def main():
    rows = []
    n_assets = 0
    n_failed = 0
    for asset_dir in sorted(ANCHORED_DIR.iterdir()):
        if not asset_dir.is_dir() or not (asset_dir / "per_window.csv").exists():
            continue
        n_assets += 1
        ticker = asset_dir.name
        try:
            pw = pd.read_csv(asset_dir / "per_window.csv", parse_dates=["test_start", "test_end"])
        except Exception:
            n_failed += 1
            continue
        for _, r in pw.iterrows():
            bh = compute_bh_sharpe(ticker, r["test_start"], r["test_end"])
            if bh is None or not np.isfinite(bh):
                continue
            rows.append({
                "asset": ticker,
                "window": int(r["window"]),
                "test_start": r["test_start"],
                "test_end": r["test_end"],
                "hmm_sharpe": float(r["test_sharpe"]),
                "bh_sharpe": bh,
                "delta": float(r["test_sharpe"]) - bh,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("ERROR: no data")
        return
    df.to_csv(OUT_DIR / "per_window_with_bh.csv", index=False)

    # Aggregate by year
    df["year"] = df["test_start"].dt.year
    by_year = df.groupby("year").agg(
        n=("delta", "count"),
        n_assets=("asset", "nunique"),
        mean_delta=("delta", "mean"),
        median_delta=("delta", "median"),
        std_delta=("delta", "std"),
        se_delta=("delta", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        mean_hmm=("hmm_sharpe", "mean"),
        mean_bh=("bh_sharpe", "mean"),
        hit_rate=("delta", lambda s: (s > 0).mean()),
    ).reset_index()
    by_year["ci_lo"] = by_year["mean_delta"] - 1.96 * by_year["se_delta"]
    by_year["ci_hi"] = by_year["mean_delta"] + 1.96 * by_year["se_delta"]
    by_year.to_csv(OUT_DIR / "by_year.csv", index=False)

    print(f"=== Alpha Decay Analysis ({n_assets} assets, {len(df)} windows) ===\n")
    print(by_year.round(3).to_string(index=False))

    # === Decay tests ===
    # 1) OLS: delta ~ year
    slope, intercept, r_val, p_val, se = stats.linregress(by_year["year"], by_year["mean_delta"])
    print(f"\n=== OLS regression (mean Δ Sharpe ~ year) ===")
    print(f"  slope  = {slope:+.4f} ΔSharpe per year")
    print(f"  intercept = {intercept:.2f}")
    print(f"  r      = {r_val:+.3f}, R² = {r_val**2:.3f}")
    print(f"  p-value = {p_val:.4g}")
    print(f"  SE     = {se:.4f}")

    # 2) Spearman on per-year mean
    rho_y, p_y = stats.spearmanr(by_year["year"], by_year["mean_delta"])
    print(f"\n=== Spearman (year vs mean Δ Sharpe, by year) ===")
    print(f"  ρ = {rho_y:+.3f}, p = {p_y:.4g}")

    # 3) Spearman on ALL individual windows (more power)
    rho_a, p_a = stats.spearmanr(df["test_start"].dt.year + df["test_start"].dt.dayofyear / 365,
                                  df["delta"])
    print(f"\n=== Spearman (year.frac vs Δ Sharpe, all windows) ===")
    print(f"  ρ = {rho_a:+.3f}, p = {p_a:.4g}  (n={len(df)})")

    # 4) Mann-Kendall on yearly mean
    def mann_kendall(x):
        from itertools import combinations
        s = sum(np.sign(xj - xi) for xi, xj in combinations(x, 2))
        n = len(x)
        var = n * (n - 1) * (2 * n + 5) / 18
        if s > 0:
            z = (s - 1) / np.sqrt(var)
        elif s < 0:
            z = (s + 1) / np.sqrt(var)
        else:
            z = 0
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        return s, z, p
    s_mk, z_mk, p_mk = mann_kendall(by_year["mean_delta"].values)
    print(f"\n=== Mann-Kendall trend test (yearly mean Δ Sharpe) ===")
    print(f"  S = {s_mk}, Z = {z_mk:+.3f}, p = {p_mk:.4g}")

    # === Plots ===
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Panel 1: Mean Δ Sharpe by year with OLS trend
    ax = axes[0, 0]
    ax.errorbar(by_year["year"], by_year["mean_delta"],
                yerr=[by_year["mean_delta"] - by_year["ci_lo"],
                      by_year["ci_hi"] - by_year["mean_delta"]],
                fmt="o-", capsize=4, color="tab:red", lw=2, ms=8,
                label="mean Δ Sharpe ± 95% CI")
    # OLS trend line
    xs = np.linspace(by_year["year"].min(), by_year["year"].max(), 50)
    ax.plot(xs, slope * xs + intercept, "k--", lw=1.5, alpha=0.7,
            label=f"OLS slope = {slope:+.3f}/year (p={p_val:.3g})")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("Test start year")
    ax.set_ylabel("Mean Δ Sharpe")
    title_decay = "DECAYING" if (slope < 0 and p_val < 0.05) else ("GROWING" if (slope > 0 and p_val < 0.05) else "STABLE")
    ax.set_title(f"α-Decay Test (yearly): {title_decay}\n"
                 f"ρ = {rho_y:+.3f}, p = {p_y:.3g}, OLS slope = {slope:+.3f}",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    # Panel 2: Sample size + hit rate by year
    ax = axes[0, 1]
    ax.bar(by_year["year"], by_year["n_assets"], color="lightgray", alpha=0.7, label="# assets")
    ax2 = ax.twinx()
    ax2.plot(by_year["year"], by_year["hit_rate"], "o-", color="tab:blue",
             lw=2, ms=8, label="hit rate")
    ax2.axhline(0.5, color="k", lw=0.7, ls=":", alpha=0.5)
    ax.set_xlabel("Test start year")
    ax.set_ylabel("# assets tested", color="gray")
    ax2.set_ylabel("Hit rate (RCVT > B&H)", color="tab:blue")
    ax2.set_ylim(0, 1.05)
    ax.set_title("Sample Size & Hit Rate by Year", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    # Panel 3: Rolling 3-year mean ΔSharpe (smoothed trend)
    ax = axes[1, 0]
    by_year_sorted = by_year.sort_values("year").reset_index(drop=True)
    by_year_sorted["rolling_mean"] = by_year_sorted["mean_delta"].rolling(3, min_periods=1).mean()
    ax.plot(by_year_sorted["year"], by_year_sorted["mean_delta"], "o-",
            color="tab:red", alpha=0.5, label="yearly mean Δ Sharpe", lw=1)
    ax.plot(by_year_sorted["year"], by_year_sorted["rolling_mean"], "o-",
            color="darkred", lw=2.5, ms=6, label="3-year rolling mean")
    ax.fill_between(by_year_sorted["year"],
                    by_year_sorted["rolling_mean"] - by_year_sorted["se_delta"],
                    by_year_sorted["rolling_mean"] + by_year_sorted["se_delta"],
                    color="darkred", alpha=0.15, label="± SE")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("Test start year")
    ax.set_ylabel("Δ Sharpe (smoothed)")
    ax.set_title("3-Year Rolling Mean — Trend Visibility", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    # Panel 4: Per-window scatter (all 5000+ windows), colored by year
    ax = axes[1, 1]
    df["year_frac"] = df["test_start"].dt.year + df["test_start"].dt.dayofyear / 365
    sc = ax.scatter(df["year_frac"], df["delta"], c=df["year_frac"],
                    cmap="viridis", alpha=0.4, s=12)
    ax.axhline(0, color="k", lw=0.5, ls=":")
    # Add OLS line
    slope2, intercept2, _, _, _ = stats.linregress(df["year_frac"], df["delta"])
    xs2 = np.linspace(df["year_frac"].min(), df["year_frac"].max(), 50)
    ax.plot(xs2, slope2 * xs2 + intercept2, "r-", lw=2,
            label=f"OLS slope = {slope2:+.4f}/year")
    ax.set_xlabel("Test start (year)")
    ax.set_ylabel("Δ Sharpe per window")
    ax.set_title(f"All {len(df)} Individual Windows\n"
                 f"OLS slope = {slope2:+.4f}/year",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, label="year")

    fig.suptitle(f"Alpha Decay Test — RCVT vs Buy-and-Hold over Time\n"
                 f"({n_assets} assets, {len(df)} walk-forward windows)",
                 fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alpha_decay.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {OUT_DIR / 'alpha_decay.png'}")
    print(f"Saved: {OUT_DIR / 'by_year.csv'}")
    print(f"Saved: {OUT_DIR / 'per_window_with_bh.csv'}")


if __name__ == "__main__":
    main()
