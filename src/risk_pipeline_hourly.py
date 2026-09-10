"""BTC 1h risk-management pipeline (adapted from the XAU daily work).

Pipeline (mirrors hmm_soft_weight_wf.py):
  1. Load BTC 1m from parquet, aggregate to 1h
  2. Feature engineering (RV on 24h window, lags 1..5, rolling stats)
  3. Train RF vol forecast on 80% training slice
  4. Walk-forward HMM soft-weight with K=3, ~3-month windows
  5. Block-bootstrap CIs vs static-soft and B&H
  6. Save plots + JSON summary

Output: results_smoke/btc_hourly/
"""
from __future__ import annotations
from itertools import product
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestRegressor

# ---------- config (hourly) ----------
DATA_DIR = Path("/home/davidv/Dokumente/Risikooptimierung/data")
TC_PER_SIDE = 0.0002            # 2 bps/side — typical Binance spot taker fee
PERIODS_PER_YEAR = 8760         # 24 × 365 (crypto 24/7)
ROLL = 24                       # 24h "daily-equivalent" realized vol window
LAGS = 5                        # hourly lag features
N_OPTUNA_TRIALS = 5
SEED = 42
K_HMM = 3
TARGET_GRID = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]  # hourly implied vol targets (sqrt-annualised)
SPLIT = 0.8
TRAIN_END_DATE = "2020-01-01"   # hard cutoff: train ends here, test starts here
WF_TRAIN_MIN = 4320             # ~180d hourly training (drawn from train fold only)
WF_TEST_SIZE = 2160             # ~90d hourly test
WF_STEP = 2160                  # 90-day step
VOL_TARGET_ANN = 0.60           # annualized vol budget for hit-metric (60% for crypto)
N_BOOT = 2000
BLOCK_SIZE = 168                # 1 week blocks
SEED_BOOT = 1234
EPS = 1e-3

OUT_DIR = Path("results_smoke/btc_hourly")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- load + resample ----------
def load_btc_1h():
    files = sorted(DATA_DIR.glob("crypto_BTC_USD_1m*.parquet"))
    print(f"Loading {len(files)} parquet files...")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values("ts").drop_duplicates("ts")
    df["Date"] = pd.to_datetime(df["ts"])
    df = df.set_index("Date")[["open", "high", "low", "close", "volume"]].astype(float)
    print(f"BTC 1m total: {len(df):,} bars, {df.index.min()} → {df.index.max()}")
    # Resample to 1h: take last close, sum volume
    h1 = df.resample("1h").agg({"open": "first", "high": "max", "low": "min",
                                 "close": "last", "volume": "sum"}).dropna(subset=["close"])
    print(f"BTC 1h: {len(h1):,} bars, {h1.index.min()} → {h1.index.max()}")
    return h1["close"]


btc = load_btc_1h()
price = btc
log_ret_full = np.log(price / price.shift(1))
log_ret_full = log_ret_full.dropna()


# ---------- features (period-aware) ----------
def build_features(close: pd.Series, window: int, lags: int) -> pd.DataFrame:
    df = pd.DataFrame({"price": close}).dropna()
    log_ret = np.log(df["price"] / df["price"].shift(1))
    df["log_return"] = log_ret
    df["rv"] = log_ret.rolling(window).std() * np.sqrt(PERIODS_PER_YEAR)  # annualized
    for lag in range(1, lags + 1):
        df[f"ret_lag_{lag}"] = log_ret.shift(lag)
        df[f"rv_lag_{lag}"] = df["rv"].shift(lag)
    df["ret_roll_mean"] = log_ret.rolling(window).mean()
    df["ret_roll_std"] = log_ret.rolling(window).std()
    df["ret_roll_skew"] = log_ret.rolling(window).skew()
    df["ret_roll_kurt"] = log_ret.rolling(window).kurt()
    df["target"] = df["rv"].shift(-1)  # next-period RV
    return df.dropna()


feat = build_features(price, window=ROLL, lags=LAGS)
print(f"Features: {feat.shape}, columns={list(feat.columns)}")

train_n = int(len(feat) * SPLIT)
train = feat.iloc[:train_n]
test = feat.iloc[train_n:]
test_idx = test.index
N_TEST = len(test_idx)
print(f"[legacy 80/20 split] train n={len(train)} ({train.index.min()} → {train.index.max()})  test n={N_TEST} ({test.index.min()} → {test.index.max()})")

# Override with hard date cutoff (much longer test fold)
train_end = pd.Timestamp(TRAIN_END_DATE, tz=feat.index.tz)
train = feat.loc[feat.index < train_end]
test = feat.loc[feat.index >= train_end]
test_idx = test.index
N_TEST = len(test_idx)
print(f"[date cutoff {TRAIN_END_DATE}] train n={len(train)} ({train.index.min()} → {train.index.max()})")
print(f"[date cutoff {TRAIN_END_DATE}] test  n={N_TEST} ({test.index.min()} → {test.index.max()}) = {N_TEST/24:.0f} days")

X_train = train.drop(columns=["target"]); y_train = train["target"]
X_test = test.drop(columns=["target"]);   y_test = test["target"]

# Train RF (small Optuna)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 80, 200),
        "max_depth": trial.suggest_int("max_depth", 5, 25),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
    }
    m = RandomForestRegressor(**params, n_jobs=-1, random_state=SEED).fit(X_train, y_train)
    p = m.predict(X_train)
    return -float(np.mean((p - y_train) ** 2))  # negative MSE for maximize


print(f"Optuna trials={N_OPTUNA_TRIALS}...")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
print(f"Best params: {study.best_params}  train MSE={-study.best_value:.6f}")

rf = RandomForestRegressor(**study.best_params, n_jobs=-1, random_state=SEED).fit(X_train, y_train)
pred = pd.Series(rf.predict(X_test), index=test_idx, name="pred")
pred_arr = pred.values

# Realized RV (annualized) on test slice
rv_realized = (log_ret_full * np.sqrt(PERIODS_PER_YEAR)).reindex(test_idx).fillna(0).values
log_ret_arr = log_ret_full.reindex(test_idx).fillna(0).values

# HMM features — STRICT backward-looking at hour t-1 (decision at end of t acts on r[t+1]).
# We compute each feature on a series shifted by 1h so that .iloc[-1] corresponds to
# log_ret[t-1] rather than log_ret[t], removing implicit contemporaneous-return leak.
rv_lag1 = pd.Series(rv_realized).shift(1).fillna(0.0).values
vol_zscore = pd.Series(rv_lag1).rolling(168).apply(
    lambda s: (s.iloc[-1] - s.mean()) / (s.std() + 1e-12), raw=False
).fillna(0.0).values
vol_of_vol = pd.Series(rv_lag1).rolling(72).std().fillna(0.0).values
vol_return = np.concatenate([[0.0], np.diff(rv_lag1)])

# Drop contemporaneous log_return — including it lets the HMM perfectly
# identify "did this hour go up/down?" and turns the strategy into momentum,
# not vol-targeting. Use lagged features only.
hmm_X = np.column_stack([vol_zscore, vol_of_vol, vol_return])
hmm_X = np.nan_to_num(hmm_X, nan=0.0, posinf=0.0, neginf=0.0)
mu = hmm_X.mean(axis=0); sd = hmm_X.std(axis=0) + 1e-12
hmm_X_std = (hmm_X - mu) / sd
first_valid = 168  # 1 week for warmup
hmm_X_eff = hmm_X_std[first_valid:]
eff_idx = test_idx[first_valid:]
print(f"HMM features: {hmm_X_eff.shape} (after dropping first {first_valid} hours = {first_valid/24:.0f} days)")


# ---------- helpers ----------
def soft_position(target_vol, pred_vals):
    return np.clip(target_vol / np.maximum(pred_vals, EPS), 0.0, 1.0)


def metrics(r, pos):
    if len(r) < 100 or r.std(ddof=1) < 1e-12:
        return {"ann_ret": 0.0, "ann_vol": 0.0, "sharpe": np.nan,
                "max_dd": 0.0, "pct_within": float("nan")}
    sd = float(r.std(ddof=1))
    ann_r = r.mean() * PERIODS_PER_YEAR
    ann_v = sd * np.sqrt(PERIODS_PER_YEAR)
    eq = np.cumsum(r); peak = np.maximum.accumulate(eq)
    max_dd = float(-(eq - peak).min())
    # per-hour returns → rolling vol on 24h window for hit-metric
    roll_v = pd.Series(r).rolling(24).std(ddof=1).dropna() * np.sqrt(PERIODS_PER_YEAR)
    pct_within = float((roll_v <= VOL_TARGET_ANN).mean()) if len(roll_v) else float("nan")
    return dict(ann_ret=ann_r, ann_vol=ann_v, sharpe=ann_r / ann_v,
                max_dd=max_dd, pct_within=pct_within)


def sharpe_rank(r):
    if len(r) < 100 or r.std(ddof=1) < 1e-12: return -np.inf
    return float(r.mean() / r.std(ddof=1))


def select_per_state_targets(tr_probs, train_logret, train_pred):
    K = tr_probs.shape[1]
    per_target_pos = {tg: soft_position(tg, train_pred) for tg in TARGET_GRID}
    best_tup, best_s = None, -np.inf
    for combo in product(TARGET_GRID, repeat=K):
        n_tr = len(tr_probs)
        mix_pos = np.zeros(n_tr)
        for k, tg in enumerate(combo):
            mix_pos += tr_probs[:, k] * per_target_pos[tg][:n_tr]
        tc = np.abs(np.diff(mix_pos, prepend=mix_pos[0])) * TC_PER_SIDE
        r = mix_pos * train_logret - tc
        s = sharpe_rank(r)
        if s > best_s:
            best_s, best_tup = s, combo
    return {k: float(best_tup[k]) for k in range(K)}, best_s


# ---------- walk-forward ----------
windows = []
w = 0
n_eff = len(eff_idx)
while True:
    te = w + WF_TRAIN_MIN
    if te + 168 >= n_eff:
        break
    fs_eff = te
    fe_eff = min(fs_eff + WF_TEST_SIZE, n_eff)
    if fe_eff - fs_eff < 720:  # at least 30 days
        break
    full_fs = first_valid + fs_eff
    full_fe = first_valid + fe_eff
    windows.append((w, te, full_fs, full_fe, fe_eff - fs_eff))
    w += WF_STEP
    if fe_eff >= n_eff:
        break

print(f"\nWalk-forward: {len(windows)} windows (train={WF_TRAIN_MIN}h, test={WF_TEST_SIZE}h, step={WF_STEP}h)")
for ww in windows:
    print(f"  train h:{eff_idx[ww[0]]} → {eff_idx[ww[1]-1]}  "
          f"test  h:{test_idx[ww[2]]} → {test_idx[ww[3]-1]}  ({ww[4]}h = {ww[4]/24:.0f}d)")

blend_pos_full = np.full(N_TEST, np.nan)
window_results = []
for w_idx, (eff0, eff1, fs, fe, n_days) in enumerate(windows):
    X_tr = hmm_X_eff[eff0:eff1]
    X_te = hmm_X_eff[eff1:(eff1 + n_days)]
    try:
        hmm = GaussianHMM(n_components=K_HMM, covariance_type="diag",
                          n_iter=200, random_state=SEED, tol=1e-4,
                          implementation="scaling").fit(X_tr)
        tr_probs = hmm.predict_proba(X_tr)
        te_probs = hmm.predict_proba(X_te) if len(X_te) > 0 else np.zeros((0, K_HMM))
    except Exception as e:
        print(f"  w{w_idx}: HMM failed ({e}); using uniform priors")
        tr_probs = np.ones((eff1 - eff0, K_HMM)) / K_HMM
        te_probs = np.ones((n_days, K_HMM)) / K_HMM

    train_logret = log_ret_arr[first_valid + eff0: first_valid + eff1]
    # FIXED: predict on the HMM training slice (test fold hours [first_valid+eff0 : first_valid+eff1])
    # so target optimization doesn't leak test-fold predictions into in-sample selection.
    train_pred = rf.predict(X_test.iloc[first_valid + eff0: first_valid + eff1])
    best_thr, train_sharpe = select_per_state_targets(tr_probs, train_logret, train_pred)

    test_orig_idx = np.arange(fs, fe)
    pos_window = np.zeros(len(test_orig_idx))
    for k in range(K_HMM):
        tg = best_thr[k]
        binary_pos = soft_position(tg, pred_arr[test_orig_idx])
        w_probs = te_probs[:, k]
        pos_window += w_probs * binary_pos

    blend_pos_full[test_orig_idx] = pos_window
    r_win = pos_window * log_ret_arr[test_orig_idx]
    tc_win = np.abs(np.diff(pos_window, prepend=pos_window[0])) * TC_PER_SIDE
    r_net = r_win - tc_win
    m = metrics(r_net, pos_window)
    window_results.append({
        "window": w_idx, "test_start": test_idx[fs], "test_end": test_idx[fe - 1],
        "best_target_per_state": best_thr, "train_sharpe": float(train_sharpe),
        "test_sharpe": m["sharpe"], "test_max_dd": m["max_dd"],
        "test_ann_ret": m["ann_ret"], "pct_invested": float(pos_window.mean()),
    })

covered = np.zeros(N_TEST, dtype=bool)
for ww in windows:
    covered[ww[2]:ww[3]] = True


def pos_to_returns(pos_full):
    r_full = np.full(N_TEST, np.nan)
    for ww in windows:
        fs, fe = ww[2], ww[3]
        pos = pos_full[fs:fe]
        r = pos * log_ret_arr[fs:fe]
        tc = np.abs(np.diff(pos, prepend=pos[0])) * TC_PER_SIDE
        r_full[fs:fe] = r - tc
    return r_full


# ---------- baselines ----------
def soft_returns_full(target_vol):
    pos = soft_position(target_vol, pred_arr)
    r = pos * log_ret_arr
    tc = np.abs(np.diff(pos, prepend=pos[0])) * TC_PER_SIDE
    return r - tc


best_t, _ = max(((tg, sharpe_rank(soft_returns_full(tg)[covered])) for tg in TARGET_GRID),
                key=lambda x: x[1])
print(f"\nBest static target_vol (in-sample on covered days): {best_t}")

bh_r = log_ret_arr.copy()
static_r = soft_returns_full(best_t)
blend_r = pos_to_returns(blend_pos_full)

m_bh = metrics(bh_r[covered], np.ones(covered.sum()))
m_static = metrics(static_r[covered], soft_position(best_t, pred_arr)[covered])
m_blend = metrics(blend_r[covered], blend_pos_full[covered])

print("\n=== Headline (n={:.0f} OOS bars = {:.0f} OOS days) ===".format(covered.sum(), covered.sum() / 24))
for name, m in [("B&H", m_bh), (f"static-soft t={best_t}", m_static), ("HMM soft-mix", m_blend)]:
    print(f"  {name:25s}: ret={m['ann_ret']:.4f}  vol={m['ann_vol']:.4f}  "
          f"sharpe={m['sharpe']:.4f}  max_dd={m['max_dd']:.4f}  pct_within={m['pct_within']:.4f}")


# ---------- block bootstrap ----------
def block_bootstrap(returns, n_boot, block, seed):
    n = len(returns); n_blocks = (n + block - 1) // block
    starts = np.arange(0, n, block)
    rng = np.random.default_rng(seed)
    out = {"sharpe": [], "max_dd": [], "pct_within": []}
    for _ in range(n_boot):
        chosen = rng.integers(0, len(starts), size=n_blocks)
        sample = np.concatenate([returns[s:min(s + block, n)] for s in starts[chosen]])[:n]
        sd = float(np.std(sample, ddof=1))
        if sd < 1e-12: continue
        sharpe = (sample.mean() * PERIODS_PER_YEAR) / (sd * np.sqrt(PERIODS_PER_YEAR))
        eq = np.cumsum(sample); peak = np.maximum.accumulate(eq)
        max_dd = float(-(eq - peak).min())
        roll_v = pd.Series(sample).rolling(24).std(ddof=1).dropna() * np.sqrt(PERIODS_PER_YEAR)
        pct_within = float((roll_v <= VOL_TARGET_ANN).mean()) if len(roll_v) else float("nan")
        out["sharpe"].append(sharpe); out["max_dd"].append(max_dd); out["pct_within"].append(pct_within)
    return out


print(f"\nBlock bootstrap N={N_BOOT}, block={BLOCK_SIZE}h ({BLOCK_SIZE/24:.0f}d)...")
boots = {n: block_bootstrap(r[covered], N_BOOT, BLOCK_SIZE, SEED_BOOT + i)
         for i, (n, r) in enumerate([("B&H", bh_r), ("static-soft", static_r), ("HMM-mix", blend_r)])}


def stats(arr):
    a = np.asarray(arr)
    return (float(a.mean()), float(np.quantile(a, 0.025)),
            float(np.median(a)), float(np.quantile(a, 0.975)))


print("\nBootstrap Sharpe (mean, 95% CI):")
for n in ["B&H", "static-soft", "HMM-mix"]:
    m, lo, _, hi = stats(boots[n]["sharpe"])
    print(f"  {n:18s}: {m:.3f}  [{lo:.3f}, {hi:.3f}]")

print("\nBootstrap Max-Drawdown (mean, 95% CI):")
for n in ["B&H", "static-soft", "HMM-mix"]:
    m, lo, _, hi = stats(boots[n]["max_dd"])
    print(f"  {n:18s}: {m:.3f}  [{lo:.3f}, {hi:.3f}]")


def paired_p(a, b, key):
    arr_a = np.asarray(boots[a][key]); arr_b = np.asarray(boots[b][key])
    n = min(len(arr_a), len(arr_b))
    return float((arr_a[:n] >= arr_b[:n]).mean())


print(f"\nP(HMM-mix Sharpe ≥ B&H)        = {paired_p('HMM-mix','B&H','sharpe'):.3f}")
print(f"P(HMM-mix Sharpe ≥ static-soft) = {paired_p('HMM-mix','static-soft','sharpe'):.3f}")

# ---------- plots ----------
fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True,
                          gridspec_kw={"height_ratios": [3, 2, 2]})
ax = axes[0]
for label, r in [("B&H (BTC 1h)", bh_r), (f"static-soft t={best_t}", static_r), ("HMM soft-mix", blend_r)]:
    cs = np.where(np.isnan(r), 0.0, np.where(np.isfinite(r), r, 0.0))
    eq = np.exp(np.cumsum(cs))
    eq = np.where(np.isnan(r), np.nan, eq)
    ax.plot(test_idx, eq, lw=0.9, label=label)
ax.set_ylabel("Equity (start=1)")
ax.set_title(f"BTC 1h Risk-Modul: B&H vs Static-Soft vs HMM ({len(windows)} WF-Fenster, "
             f"{int(covered.sum()/24)} OOS-Tage, Block-Bootstrap N={N_BOOT})")
ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
for label, r, c in [("B&H", bh_r, "tab:blue"), ("static-soft", static_r, "tab:orange"),
                     ("HMM-mix", blend_r, "tab:green")]:
    # rolling vol on 24h window annualized
    rv = pd.Series(r).rolling(24).std(ddof=1).fillna(0) * np.sqrt(PERIODS_PER_YEAR)
    ax.plot(test_idx, rv, lw=0.9, color=c, label=label)
ax.axhline(VOL_TARGET_ANN, color="k", lw=1, ls="--", label=f"Vol-Target {VOL_TARGET_ANN*100:.0f}%")
ax.set_ylabel("Rolling 24h ann. Vol"); ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)

ax = axes[2]
for label, r, c in [("B&H", bh_r, "tab:blue"), ("static-soft", static_r, "tab:orange"),
                     ("HMM-mix", blend_r, "tab:green")]:
    cs = np.cumsum(pd.Series(r).fillna(0))
    peak = np.maximum.accumulate(cs)
    dd = cs - peak
    ax.fill_between(test_idx, dd, 0, alpha=0.25, color=c, label=label)
ax.set_ylabel("Drawdown (log)"); ax.set_xlabel("Datum")
ax.legend(loc="lower left", fontsize=9, ncol=3); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "btc_hourly_equity.png", dpi=130)
plt.close(fig)

# Per-window
fig, ax = plt.subplots(figsize=(13, 4))
xs = np.arange(len(window_results))
sharpes = [w["test_sharpe"] for w in window_results]
inv = [w["pct_invested"] for w in window_results]
ax2 = ax.twinx()
ax.bar(xs, sharpes, color="tab:green", alpha=0.75, label="OOS Sharpe")
ax2.plot(xs, inv, "o-", color="tab:blue", label="% invested")
ax.axhline(0, color="k", lw=1); ax2.set_ylim(0, 1.1)
ax.set_xticks(xs)
ax.set_xticklabels(
    [f"{w['test_start'].date()}\n→ {w['test_end'].date()}\n{w['test_sharpe']:+.2f}"
     for w in window_results], fontsize=7)
ax.set_ylabel("OOS Sharpe"); ax.set_title("Per-Window OOS Sharpe (BTC 1h HMM)")
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT_DIR / "btc_hourly_per_window.png", dpi=130)
plt.close(fig)

# Bootstrap distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, key in zip(axes, ["sharpe", "max_dd"]):
    dist = [boots[n][key] for n in ["B&H", "static-soft", "HMM-mix"]]
    bp = ax.boxplot(dist, tick_labels=["B&H", "static-soft", "HMM-mix"],
                    patch_artist=True, showmeans=True, meanline=True,
                    boxprops=dict(facecolor="lightblue", alpha=0.5))
    bp["boxes"][2].set_facecolor("lightgreen")
    ax.set_title(key); ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT_DIR / "btc_hourly_bootstrap.png", dpi=130)
plt.close(fig)

# ---------- save ----------
out = {
    "best_params": study.best_params,
    "best_static_target": best_t,
    "covered_bars": int(covered.sum()),
    "covered_days": float(covered.sum() / 24),
    "n_windows": len(windows),
    "headline": {n: {k: round(v, 6) if isinstance(v, float) else v for k, v in m.items()}
                  for n, m in [("B&H", m_bh), (f"static_soft_t{best_t}", m_static),
                                ("HMM_soft_mix", m_blend)]},
    "per_window": [{
        "window": r["window"],
        "test_start": r["test_start"].isoformat(),
        "test_end": r["test_end"].isoformat(),
        "best_target_per_state": r["best_target_per_state"],
        "train_sharpe": round(r["train_sharpe"], 4),
        "test_sharpe": round(r["test_sharpe"], 4) if r["test_sharpe"] == r["test_sharpe"] else None,
        "pct_invested": round(r["pct_invested"], 4),
    } for r in window_results],
}
(OUT_DIR / "summary.json").write_text(json.dumps(out, indent=2, default=str))
pd.DataFrame(window_results).to_csv(OUT_DIR / "per_window.csv", index=False)

print(f"\nSaved to {OUT_DIR}/")
for f in sorted(OUT_DIR.glob("*")):
    print(" -", f, f.stat().st_size, "bytes")
