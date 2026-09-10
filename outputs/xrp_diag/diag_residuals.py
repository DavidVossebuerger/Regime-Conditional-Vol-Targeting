"""Residual diagnostic for XAU/USD RF vol forecast.
Uses VolatilityBacktester's own load_data + prepare_features to keep the
prep 100% identical to the smoke run. Then re-fits RF with logged best params
and plots scatter / residuals-vs-predicted / Q-Q + histogram.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

from volatility_backtester import VolatilityBacktester

OUT_DIR = Path("results_smoke/diag")
OUT_DIR.mkdir(parents=True, exist_ok=True)

vb = VolatilityBacktester(data_dir="../Kursdaten", results_dir="results_smoke/_unused")
data = vb.load_data(pattern="xauusd-d1")
series = data.iloc[:, 0]

# prep identical to run_full_backtest.py
feat = vb.prepare_features(series, window=20, lags=5)
train, test = vb.train_test_split(feat, split=0.8)
X_train = train.drop(columns=["target"])
y_train = train["target"]
X_test = test.drop(columns=["target"])
y_test = test["target"]

print(f"train n={len(X_train)}  test n={len(X_test)}  features={list(X_train.columns)}")

# logged best Optuna params (smoke run)
model = RandomForestRegressor(
    n_estimators=196, max_depth=19, max_features="log2", n_jobs=-1, random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_true = y_test.values
resid = y_true - y_pred

rmse = float(np.sqrt(np.mean(resid**2)))
mae = float(np.mean(np.abs(resid)))
r2 = float(1 - np.sum(resid**2) / np.sum((y_true - y_true.mean()) ** 2))
mape = float(np.mean(np.abs(resid / y_true)) * 100)
print(f"Test fold: RMSE={rmse:.6f}  MAE={mae:.6f}  R²={r2:.4f}  MAPE={mape:.2f}%")

abs_r = np.abs(resid)
q = np.quantile(abs_r, [0.5, 0.9, 0.95, 0.99, 0.999, 1.0])
print(f"|resid| quantiles 50/90/95/99/99.9/100% = " +
      " / ".join(f"{v:.5f}" for v in q))

sse = resid**2
order = np.argsort(-sse)
print(f"SSE share top1={sse[order[:1]].sum()/sse.sum():.1%}, "
      f"top5={sse[order[:5]].sum()/sse.sum():.1%}, "
      f"top10={sse[order[:10]].sum()/sse.sum():.1%}, "
      f"top50={sse[order[:50]].sum()/sse.sum():.1%}, "
      f"top1%=  {sse[order[:int(0.01*len(sse))]].sum()/sse.sum():.1%}")

# dates for the test fold
idx = y_test.index

# --- 1) scatter predicted vs realized ---
fig, ax = plt.subplots(figsize=(7, 7))
mx = max(y_true.max(), y_pred.max()) * 1.05
lims = [0, mx]
ax.scatter(y_pred, y_true, s=10, alpha=0.35, color="#1f77b4", label="test points")
ax.plot(lims, lims, "k--", lw=1, label="identity")
m, b = np.polyfit(y_pred, y_true, 1)
xx = np.linspace(*lims, 50)
ax.plot(xx, m * xx + b, color="#d62728", lw=1.2,
        label=f"OLS fit  y = {m:.2f}·x {b:+.4f}")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Predicted realized vol")
ax.set_ylabel("Realized vol (next-day RV)")
ax.set_title(f"Predicted vs Realized — XAU d1, RF (R²={r2:.3f}, n={len(resid)})")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "scatter_pred_vs_realized.png", dpi=130)
plt.close(fig)

# --- 2) residuals vs predicted (heteroskedasticity) ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_pred, resid, s=10, alpha=0.4, color="#2ca02c")
ax.axhline(0, color="k", lw=1)
order_p = np.argsort(y_pred)
window_n = max(50, len(resid) // 30)
roll_abs = pd.Series(np.abs(resid[order_p])).rolling(window_n, center=True).mean()
ax.plot(y_pred[order_p], roll_abs, color="#d62728", lw=1.5,
        label=f"rolling |resid| (n={window_n})")
ax.set_xlabel("Predicted vol")
ax.set_ylabel("Residual = realized − predicted")
ax.set_title("Heteroskedastizität — Fehler vs Predicted")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "residuals_vs_predicted.png", dpi=130)
plt.close(fig)

# --- 3) Q-Q + Histogram/KDE ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
stats.probplot(resid, dist="norm", plot=axes[0])
axes[0].get_lines()[1].set_color("#d62728"); axes[0].get_lines()[1].set_lw(1.5)
axes[0].set_title("Q-Q: Residuen ~ N(0,σ)")
axes[0].set_xlabel("Theoretische Quantile"); axes[0].set_ylabel("Sample-Quantile")
axes[0].grid(alpha=0.3)

ax = axes[1]
ax.hist(resid, bins=60, density=True, alpha=0.6, color="#1f77b4", edgecolor="white")
xs = np.linspace(resid.min(), resid.max(), 300)
ax.plot(xs, stats.norm.pdf(xs, loc=resid.mean(), scale=resid.std()),
        color="#d62728", lw=2,
        label=f"N(μ={resid.mean():.4f}, σ={resid.std():.4f})")
kde = stats.gaussian_kde(resid)
ax.plot(xs, kde(xs), color="#2ca02c", lw=2, ls="--", label="KDE")
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("Residuum"); ax.set_ylabel("Dichte")
ax.set_title(
    f"Residuen (n={len(resid)}, skew={stats.skew(resid):.2f}, kurt={stats.kurtosis(resid):.2f})"
)
ax.legend(); ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR / "qq_and_histogram.png", dpi=130)
plt.close(fig)

print("done →", OUT_DIR)
for f in sorted(OUT_DIR.glob("*.png")):
    print("  -", f, f.stat().st_size, "bytes")
