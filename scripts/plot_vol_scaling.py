"""Plot vol-scaling experiment results — comprehensive view.

Generates:
- outputs/vol_scaling/edge_by_vol.png — multi-panel summary
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT_DIR = Path("outputs/vol_scaling")
df = pd.read_csv(OUT_DIR / "summary.csv")

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# Panel 1: Mean ΔSharpe by scale (with bootstrap CI)
ax = axes[0, 0]
agg = df.groupby("scale").agg(
    mean=("delta_sharpe", "mean"),
    std=("delta_sharpe", "std"),
    n=("asset", "count"),
).reset_index()
# Bootstrap 95% CI per scale
rng = np.random.default_rng(42)
ci_lo, ci_hi = [], []
for _, row in agg.iterrows():
    sub = df[df["scale"] == row["scale"]]["delta_sharpe"].values
    boots = [np.mean(rng.choice(sub, size=len(sub), replace=True)) for _ in range(2000)]
    ci_lo.append(np.percentile(boots, 2.5))
    ci_hi.append(np.percentile(boots, 97.5))
agg["ci_lo"] = ci_lo
agg["ci_hi"] = ci_hi

ax.errorbar(agg["scale"], agg["mean"],
            yerr=[agg["mean"] - agg["ci_lo"], agg["ci_hi"] - agg["mean"]],
            fmt="o-", capsize=5, color="tab:red", lw=2.5, ms=10,
            label="mean ± 95% bootstrap CI")
ax.plot(agg["scale"], agg["mean"], "o", color="darkred", ms=12, zorder=10)
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.set_xscale("log")
ax.set_xlabel("Vol scale (×, log scale)", fontsize=11)
ax.set_ylabel("Δ Sharpe (HMM − B&H)", fontsize=11)
ax.set_title(f"Mean Δ Sharpe vs Vol Scale\nSpearman ρ = +0.584, p = 2.3×10⁻²³ (n={len(df)})",
             fontsize=12, fontweight="bold")
ax.grid(alpha=0.3, which="both")
ax.legend(loc="upper left")

# Panel 2: Hit rate by scale
ax = axes[0, 1]
hit_rates = df.groupby("scale").apply(
    lambda g: (g["delta_sharpe"] > 0).mean(), include_groups=False
).reset_index(name="hit_rate")
# Wilson 95% CI for hit rate
n_per_scale = df.groupby("scale").size().values
hit_arr = hit_rates["hit_rate"].values
from scipy.stats import beta
ci_lo_hit = []
ci_hi_hit = []
for k, n in enumerate(n_per_scale):
    successes = int(round(hit_arr[k] * n))
    lo, hi = beta.interval(0.95, successes + 0.5, n - successes + 0.5)
    ci_lo_hit.append(lo); ci_hi_hit.append(hi)
hit_rates["ci_lo"] = ci_lo_hit
hit_rates["ci_hi"] = ci_hi_hit

ax.errorbar(hit_rates["scale"], hit_rates["hit_rate"],
            yerr=[hit_rates["hit_rate"] - hit_rates["ci_lo"],
                  hit_rates["ci_hi"] - hit_rates["hit_rate"]],
            fmt="s-", capsize=5, color="tab:blue", lw=2.5, ms=10,
            label="hit rate ± 95% Wilson CI")
ax.axhline(0.5, color="k", lw=0.7, ls=":", label="50% null")
ax.set_xscale("log")
ax.set_xlabel("Vol scale (×, log scale)", fontsize=11)
ax.set_ylabel("Hit rate (RCVT > B&H)", fontsize=11)
ax.set_title("Hit Rate by Vol Scale\n(climbs 23% → 90% as vol rises)",
             fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3, which="both")
ax.legend(loc="lower right")

# Panel 3: Scatter of all 240 observations, colored by scale
ax = axes[1, 0]
scales = sorted(df["scale"].unique())
colors = plt.cm.viridis(np.linspace(0, 1, len(scales)))
for i, sc in enumerate(scales):
    sub = df[df["scale"] == sc]
    ax.scatter([sc] * len(sub), sub["delta_sharpe"], color=colors[i],
               alpha=0.6, s=35, label=f"{sc}× (n={len(sub)})", zorder=3)
# Mean line
ax.plot(agg["scale"], agg["mean"], "k-o", lw=2.5, ms=8, zorder=5,
        label="mean per scale")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.set_xscale("log")
ax.set_xlabel("Vol scale (×, log scale)", fontsize=11)
ax.set_ylabel("Δ Sharpe (per asset)", fontsize=11)
ax.set_title("All 240 Observations: Δ Sharpe by Vol Scale\n(jitter applied)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=8, loc="upper left", ncol=2)
ax.grid(alpha=0.3, which="both")

# Panel 4: Within-asset z-score vs scale (controls for cross-asset level)
ax = axes[1, 1]
df["delta_z"] = df.groupby("asset")["delta_sharpe"].transform(
    lambda s: (s - s.mean()) / s.std() if s.std() > 0 else 0.0)
# Mean z-score per scale
z_agg = df.groupby("scale")["delta_z"].mean().reset_index(name="z_mean")
ax.plot(z_agg["scale"], z_agg["z_mean"], "o-", color="darkgreen", lw=2.5, ms=10)
ax.axhline(0, color="k", lw=0.5, ls="--")
# Compute within-asset Spearman for the title
rho_within, p_within = stats.spearmanr(df["scale"], df["delta_z"])
ax.fill_between([0.2, 4.5], -0.1, 0.1, alpha=0.15, color="gray", zorder=0)
ax.set_xscale("log")
ax.set_xlabel("Vol scale (×, log scale)", fontsize=11)
ax.set_ylabel("Δ Sharpe (within-asset z-score)", fontsize=11)
ax.set_title(f"Within-Asset Standardization\nSpearman ρ = +{rho_within:.3f}, p = {p_within:.2e}",
             fontsize=12, fontweight="bold")
ax.grid(alpha=0.3, which="both")

fig.suptitle("RCVT Edge vs Underlying Volatility\n"
             f"(240 paired observations across 30 assets × 8 vol scales)",
             fontsize=14, fontweight="bold", y=1.00)
fig.tight_layout()
fig.savefig(OUT_DIR / "edge_by_vol.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / 'edge_by_vol.png'}")
