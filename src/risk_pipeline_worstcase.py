"""BTC 1h risk-modul with thermodynamic worst-case (model-uncertainty) feature.

Pipeline (extends btc_hourly_risk.py):
  1. Calibration snippet: random 3-month slice from TRAIN fold only (no test leakage)
  2. Compute V_wc(η) = -η⁻¹ log( (1/N) Σ exp(-η r_i) ) for η grid
  3. Pick η from snippet so median worstcase_ratio ≈ 1.3-1.5
  4. Build worstcase_ratio(t) over full test fold (rolling 168h window)
  5. Add as 4th HMM feature (alongside vol_zscore, vol_of_vol, vol_return)
  6. Multiplier: pos_final = pos_soft · clip(1/worstcase_ratio, 0.4, 1.0)
  7. Walk-forward + block-bootstrap, three HMM variants:
       - HMM-no-wc (baseline from before)
       - HMM-wc-feat  (HMM gets wc as feature, no multiplier)
       - HMM-wc-multi (HMM gets wc as feature + multiplier on position)
  8. Post-mortem: compare calibration snippet distribution vs OOS test fold distribution
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

# ---------- config ----------
DATA_DIR = Path("/home/davidv/Dokumente/Risikooptimierung/data")
TC_PER_SIDE = 0.0002
PERIODS_PER_YEAR = 8760
ROLL = 24
LAGS = 5
N_OPTUNA_TRIALS = 5
SEED = 42
K_HMM = 3
TARGET_GRID = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
TRAIN_END_DATE = "2020-01-01"
WF_TRAIN_MIN = 4320
WF_TEST_SIZE = 2160
WF_STEP = 2160
VOL_TARGET_ANN = 0.60
N_BOOT = 2000
BLOCK_SIZE = 168
EPS = 1e-3
# calibration snippet
CAL_SNIPPET_HOURS = 90 * 24    # 3 months of hourly bars
CAL_SEED = 7                   # reproducible snippet selection
# η grid for calibration (in nats)
ETA_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
WC_WINDOW = 168                # rolling window for worstcase ratio (1 week)
WC_FLOOR = 0.4                 # multiplicative floor on (1/wc_ratio)

OUT_DIR = Path("results_smoke/btc_worstcase")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- load BTC ----------
def load_btc_1h():
    files = sorted(DATA_DIR.glob("crypto_BTC_USD_1m*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values("ts").drop_duplicates("ts")
    df["Date"] = pd.to_datetime(df["ts"])
    df = df.set_index("Date")[["open", "high", "low", "close", "volume"]].astype(float)
    h1 = df.resample("1h").agg({"close": "last"}).dropna(subset=["close"])
    return h1["close"]


btc = load_btc_1h()
log_ret_full = np.log(btc / btc.shift(1)).dropna()


def build_features(close, window, lags):
    df = pd.DataFrame({"price": close}).dropna()
    log_ret = np.log(df["price"] / df["price"].shift(1))
    df["log_return"] = log_ret
    df["rv"] = log_ret.rolling(window).std() * np.sqrt(PERIODS_PER_YEAR)
    for lag in range(1, lags + 1):
        df[f"ret_lag_{lag}"] = log_ret.shift(lag)
        df[f"rv_lag_{lag}"] = df["rv"].shift(lag)
    df["ret_roll_mean"] = log_ret.rolling(window).mean()
    df["ret_roll_std"] = log_ret.rolling(window).std()
    df["ret_roll_skew"] = log_ret.rolling(window).skew()
    df["ret_roll_kurt"] = log_ret.rolling(window).kurt()
    df["target"] = df["rv"].shift(-1)
    return df.dropna()


feat = build_features(btc, window=ROLL, lags=LAGS)
train_end = pd.Timestamp(TRAIN_END_DATE, tz=feat.index.tz)
train = feat.loc[feat.index < train_end]
test = feat.loc[feat.index >= train_end]
test_idx = test.index
N_TEST = len(test_idx)
print(f"train n={len(train)} ({train.index.min()} → {train.index.max()})")
print(f"test  n={N_TEST} ({test.index.min()} → {test.index.max()}) = {N_TEST/24:.0f} days")

X_train = train.drop(columns=["target"]); y_train = train["target"]
X_test = test.drop(columns=["target"]);   y_test = test["target"]

# ---------- RF ----------
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
    return -float(np.mean((m.predict(X_train) - y_train) ** 2))


print(f"Optuna trials={N_OPTUNA_TRIALS}...")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
print(f"Best params: {study.best_params}")
rf = RandomForestRegressor(**study.best_params, n_jobs=-1, random_state=SEED).fit(X_train, y_train)
pred = pd.Series(rf.predict(X_test), index=test_idx, name="pred")
pred_arr = pred.values
log_ret_arr = log_ret_full.reindex(test_idx).fillna(0).values

# ============================================================
# CALIBRATION: random 3-month snippet from TRAIN fold only
# ============================================================
print(f"\n=== Calibration snippet (3 months, random from train fold) ===")
rng = np.random.default_rng(CAL_SEED)
train_idx_arr = np.arange(len(train))
# choose a random contiguous 3-month slice from the train fold
max_start = len(train) - CAL_SNIPPET_HOURS - 1
if max_start <= 0:
    raise RuntimeError("Train fold too short for calibration snippet")
cal_start = int(rng.integers(0, max_start))
cal_end = cal_start + CAL_SNIPPET_HOURS
cal_slice = train.iloc[cal_start:cal_end]
cal_logret = cal_slice["log_return"].values
print(f"calibration slice dates: {cal_slice.index.min()} → {cal_slice.index.max()}")
print(f"n={len(cal_logret)} hourly bars, mean={cal_logret.mean():.5f}, std={cal_logret.std():.5f}")

# compute V_wc for η grid on calibration snippet
def v_wc(logret: np.ndarray, eta: float) -> float:
    """Worst-case expected log-return under entropic budget η (Hansen-Sargent dual):
       V_wc(η) = η⁻¹ · log( (1/N) Σ exp(η r_i) )

    Convex duality ensures V_wc ≥ E[r]. Larger V_wc ⇒ worse expected outcome under
    model perturbation of size η. Equivalently: smaller (more negative) ⇒ safer.
    """
    z = np.log(np.mean(np.exp(eta * logret)))
    return float(z / eta)


def mean_logret(logret: np.ndarray) -> float:
    return float(np.mean(logret))


meanL = mean_logret(cal_logret)
print(f"plain mean log-return V={meanL:.5f}")
print(f"\nη-scan on calibration snippet:")
print(f"{'eta':>8s}  {'V_wc':>10s}  {'V_wc - V':>12s}  {'|V_wc|':>10s}")
eta_calibration = None
for eta in ETA_GRID:
    wc = v_wc(cal_logret, eta)
    delta = wc - meanL
    print(f"  {eta:7.3f}  {wc:10.5f}  {delta:+12.6f}  {abs(wc):10.5f}")

# Choose η: smallest one where |V_wc - V| ≈ 50% of |V|, i.e., η where worst-case is
# meaningfully worse than nominal. Avoids both trivial (η=0) and extreme (η=∞).
target_drag_pct = 0.5    # pick η such that V_wc deviates ≥50% from V (in magnitude)
chosen_eta = None
for eta in ETA_GRID:
    wc = v_wc(cal_logret, eta)
    drag = abs(wc - meanL) / max(abs(meanL), 1e-6)
    if drag >= target_drag_pct:
        chosen_eta = eta
        break
if chosen_eta is None:
    chosen_eta = ETA_GRID[-1]
eta_calibration = chosen_eta
print(f"\n→ Chosen η = {eta_calibration} (|V_wc-V|/|V| ≈ "
      f"{abs(v_wc(cal_logret, eta_calibration)-meanL)/max(abs(meanL),1e-6):.2f} on snippet)")


# ---------- worstcase_ratio over full test fold ----------
def wc_series(logret: np.ndarray, eta: float, window: int) -> np.ndarray:
    """Rolling V_wc(η; rolling window). Lag-1 internally to avoid contemporaneous leak."""
    out = np.full(len(logret), np.nan)
    for i in range(window, len(logret)):
        win = logret[i - window:i]
        out[i] = v_wc(win, eta)
    return out


wc_full = wc_series(log_ret_arr, eta_calibration, WC_WINDOW)
# shift by 1 so values at index i reflect data ending at i-1 (decision at end of hour i acts on r[i+1])
wc_test = pd.Series(wc_full, index=test_idx).shift(1).fillna(0.0).values
print(f"wc summary on test fold: median={np.nanmedian(wc_full):.6f}, "
      f"95%={np.nanquantile(wc_full, 0.95):.6f}, max={np.nanmax(wc_full):.6f}")

# ---------- HMM features (with optional wc) ----------
rv_realized = (log_ret_full * np.sqrt(PERIODS_PER_YEAR)).reindex(test_idx).fillna(0).values
rv_lag1 = pd.Series(rv_realized).shift(1).fillna(0.0).values
vol_zscore = pd.Series(rv_lag1).rolling(168).apply(
    lambda s: (s.iloc[-1] - s.mean()) / (s.std() + 1e-12), raw=False
).fillna(0.0).values
vol_of_vol = pd.Series(rv_lag1).rolling(72).std().fillna(0.0).values
vol_return = np.concatenate([[0.0], np.diff(rv_lag1)])

def build_hmm(use_wc: bool):
    cols = [vol_zscore, vol_of_vol, vol_return]
    if use_wc:
        # wc_test is already lag-1-shifted, standardised inside
        cols.append(wc_test)
    X = np.column_stack(cols)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mu = X.mean(axis=0); sd = X.std(axis=0) + 1e-12
    return (X - mu) / sd

hmm_X_nowc = build_hmm(False)
hmm_X_wc = build_hmm(True)
first_valid = 168
hmm_X_nowc_eff = hmm_X_nowc[first_valid:]
hmm_X_wc_eff = hmm_X_wc[first_valid:]
eff_idx = test_idx[first_valid:]


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
        if s > best_s: best_s, best_tup = s, combo
    return {k: float(best_tup[k]) for k in range(K)}, best_s


# ---------- walk-forward for one HMM setup + multiplier ----------
def run_walk_forward(hmm_X_eff, tag, use_multiplier: bool):
    n_eff = len(eff_idx)
    windows = []
    w = 0
    while True:
        te = w + WF_TRAIN_MIN
        if te + 168 >= n_eff: break
        fs_eff = te
        fe_eff = min(fs_eff + WF_TEST_SIZE, n_eff)
        if fe_eff - fs_eff < 720: break
        windows.append((w, te, first_valid + fs_eff, first_valid + fe_eff, fe_eff - fs_eff))
        w += WF_STEP
        if fe_eff >= n_eff: break

    print(f"\n[{tag}] {len(windows)} windows (multiplier={use_multiplier})")
    blend_pos_full = np.full(N_TEST, np.nan)
    win_results = []
    for w_idx, (eff0, eff1, fs, fe, n_days) in enumerate(windows):
        X_tr = hmm_X_eff[eff0:eff1]
        X_te = hmm_X_eff[eff1:(eff1 + n_days)]
        try:
            hmm = GaussianHMM(n_components=K_HMM, covariance_type="diag",
                              n_iter=200, random_state=SEED, tol=1e-4,
                              implementation="log").fit(X_tr)
            tr_probs = hmm.predict_proba(X_tr)
            te_probs = hmm.predict_proba(X_te) if len(X_te) > 0 else np.zeros((0, K_HMM))
        except Exception as e:
            print(f"  w{w_idx}: HMM failed: {e}")
            tr_probs = np.ones((eff1 - eff0, K_HMM)) / K_HMM
            te_probs = np.ones((n_days, K_HMM)) / K_HMM

        train_logret = log_ret_arr[first_valid + eff0: first_valid + eff1]
        train_pred = rf.predict(X_test.iloc[first_valid + eff0: first_valid + eff1])
        best_thr, train_sharpe = select_per_state_targets(tr_probs, train_logret, train_pred)

        test_orig_idx = np.arange(fs, fe)
        pos_window = np.zeros(len(test_orig_idx))
        for k in range(K_HMM):
            tg = best_thr[k]
            binary_pos = soft_position(tg, pred_arr[test_orig_idx])
            pos_window += te_probs[:, k] * binary_pos

        if use_multiplier:
            wc_window = wc_full[test_orig_idx]
            # Risk-budget multiplier: scale position down when V_wc exceeds a budget.
            # Budget = 0.1% worst-case expected hourly loss (≈ 25% annualised).
            WC_BUDGET = 0.001
            excess = np.maximum(wc_window, 0.0)
            mult = np.clip(WC_BUDGET / np.maximum(excess, 1e-6), WC_FLOOR, 1.0)
            # when V_wc is negative (worst-case still profit) → excess=0 → mult=1.0
            pos_window = pos_window * mult

        blend_pos_full[test_orig_idx] = pos_window
        r_win = pos_window * log_ret_arr[test_orig_idx]
        tc_win = np.abs(np.diff(pos_window, prepend=pos_window[0])) * TC_PER_SIDE
        r_net = r_win - tc_win
        m = metrics(r_net, pos_window)
        win_results.append({
            "window": w_idx, "test_start": test_idx[fs], "test_end": test_idx[fe - 1],
            "best_target_per_state": best_thr,
            "test_sharpe": m["sharpe"], "test_max_dd": m["max_dd"],
            "pct_invested": float(pos_window.mean()),
        })
    return blend_pos_full, windows, win_results


def pos_to_returns(pos_full, windows):
    r_full = np.full(N_TEST, np.nan)
    for ww in windows:
        fs, fe = ww[2], ww[3]
        pos = pos_full[fs:fe]
        r = pos * log_ret_arr[fs:fe]
        tc = np.abs(np.diff(pos, prepend=pos[0])) * TC_PER_SIDE
        r_full[fs:fe] = r - tc
    return r_full


def soft_returns_full(target_vol):
    pos = soft_position(target_vol, pred_arr)
    r = pos * log_ret_arr
    tc = np.abs(np.diff(pos, prepend=pos[0])) * TC_PER_SIDE
    return r - tc


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


def stats(arr):
    a = np.asarray(arr)
    return (float(a.mean()), float(np.quantile(a, 0.025)),
            float(np.median(a)), float(np.quantile(a, 0.975)))


def paired_p(a, b, key):
    arr_a = np.asarray(boots[a][key]); arr_b = np.asarray(boots[b][key])
    n = min(len(arr_a), len(arr_b))
    return float((arr_a[:n] >= arr_b[:n]).mean())


# ---------- run all variants ----------
# Coverage mask
windows_dummy = []
w = 0
while True:
    te = w + WF_TRAIN_MIN
    if te + 168 >= len(eff_idx): break
    fs_eff = te
    fe_eff = min(fs_eff + WF_TEST_SIZE, len(eff_idx))
    if fe_eff - fs_eff < 720: break
    windows_dummy.append((first_valid + fs_eff, first_valid + fe_eff))
    w += WF_STEP
    if fe_eff >= len(eff_idx): break

covered = np.zeros(N_TEST, dtype=bool)
for fs, fe in windows_dummy:
    covered[fs:fe] = True

best_t = max(TARGET_GRID, key=lambda tg: sharpe_rank(soft_returns_full(tg)[covered]))
print(f"\nBest static target_vol: {best_t}")

bh_r = log_ret_arr.copy()
static_r = soft_returns_full(best_t)

print("\n=== Run HMM-no-wc (baseline) ===")
blend_nowc_pos, windows, win_nowc = run_walk_forward(hmm_X_nowc_eff, "no-wc", False)
print("\n=== Run HMM-wc-feat (worstcase as feature only) ===")
blend_wcfeat_pos, _, win_wcfeat = run_walk_forward(hmm_X_wc_eff, "wc-feat", False)
print("\n=== Run HMM-wc-multi (feature + multiplier) ===")
blend_wcmult_pos, _, win_wcmult = run_walk_forward(hmm_X_wc_eff, "wc-multi", True)

blend_nowc_r = pos_to_returns(blend_nowc_pos, windows)
blend_wcfeat_r = pos_to_returns(blend_wcfeat_pos, windows)
blend_wcmult_r = pos_to_returns(blend_wcmult_pos, windows)

m_bh = metrics(bh_r[covered], np.ones(covered.sum()))
m_static = metrics(static_r[covered], soft_position(best_t, pred_arr)[covered])
m_nowc = metrics(blend_nowc_r[covered], blend_nowc_pos[covered])
m_wcfeat = metrics(blend_wcfeat_r[covered], blend_wcfeat_pos[covered])
m_wcmult = metrics(blend_wcmult_r[covered], blend_wcmult_pos[covered])

print(f"\n=== Headline (n={int(covered.sum()/24)} OOS days) ===")
for name, m in [("B&H", m_bh), (f"static t={best_t}", m_static),
                  ("HMM no-wc", m_nowc), ("HMM wc-feat", m_wcfeat),
                  ("HMM wc-multi", m_wcmult)]:
    print(f"  {name:18s}: ret={m['ann_ret']:.4f}  vol={m['ann_vol']:.4f}  "
          f"sharpe={m['sharpe']:.4f}  max_dd={m['max_dd']:.4f}  pct_within={m['pct_within']:.4f}")

print(f"\nBlock bootstrap N={N_BOOT}...")
boots = {n: block_bootstrap(r[covered], N_BOOT, BLOCK_SIZE, SEED + 100 + i)
         for i, (n, r) in enumerate([
             ("B&H", bh_r), ("static-soft", static_r),
             ("HMM-no-wc", blend_nowc_r),
             ("HMM-wc-feat", blend_wcfeat_r),
             ("HMM-wc-multi", blend_wcmult_r)])}

print("\nBootstrap Sharpe (mean, 95% CI):")
for n in ["B&H", "static-soft", "HMM-no-wc", "HMM-wc-feat", "HMM-wc-multi"]:
    m, lo, _, hi = stats(boots[n]["sharpe"])
    print(f"  {n:18s}: {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

print("\nBootstrap Max-Drawdown (mean, 95% CI):")
for n in ["B&H", "static-soft", "HMM-no-wc", "HMM-wc-feat", "HMM-wc-multi"]:
    m, lo, _, hi = stats(boots[n]["max_dd"])
    print(f"  {n:18s}: {m:.3f}  [{lo:.3f}, {hi:.3f}]")

p_wcmult_vs_nowc = paired_p("HMM-wc-multi", "HMM-no-wc", "sharpe")
p_wcmult_vs_bh = paired_p("HMM-wc-multi", "B&H", "sharpe")
p_wcmult_vs_static = paired_p("HMM-wc-multi", "static-soft", "sharpe")
print(f"\nP(HMM-wc-multi Sharpe ≥ HMM-no-wc) = {p_wcmult_vs_nowc:.3f}")
print(f"P(HMM-wc-multi Sharpe ≥ B&H)        = {p_wcmult_vs_bh:.3f}")
print(f"P(HMM-wc-multi Sharpe ≥ static-soft)= {p_wcmult_vs_static:.3f}")

# ============================================================
# POST-MORTEM: is calibration snippet representative of OOS?
# ============================================================
print("\n" + "=" * 60)
print("POST-MORTEM: calibration snippet vs OOS test fold")
print("=" * 60)
oos_logret = log_ret_arr[covered]
snippet_logret = cal_logret


def dist_stats(arr, name):
    a = np.asarray(arr)
    print(f"  {name:18s}: mean={a.mean():+.6f}  std={a.std():.6f}  skew={pd.Series(a).skew():+.3f}"
          f"  kurt={pd.Series(a).kurt():.3f}  VaR5={np.quantile(a,0.05):+.6f}"
          f"  CVaR5={a[a<=np.quantile(a,0.05)].mean():+.6f}"
          f"  |r|>2σ frac={(np.abs(a) > 2 * a.std()).mean():.3%}")
    # also wc_ratio summary
    wc = wc_series(a, eta_calibration, WC_WINDOW)
    print(f"  {name:18s}: V_wc median={np.nanmedian(wc):+.6f}  p95={np.nanquantile(wc,0.95):+.6f}")


dist_stats(snippet_logret, "calibration")
dist_stats(oos_logret, "OOS-test-fold")

# KS test on distributions
from scipy.stats import ks_2samp
ks_stat, ks_p = ks_2samp(snippet_logret, oos_logret)
print(f"\n  KS-test snippet vs OOS: stat={ks_stat:.4f}  p-value={ks_p:.4f}")
print(f"  (if p > 0.05 → no statistical evidence distributions differ)")

# ============================================================
# plots
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True,
                          gridspec_kw={"height_ratios": [3, 2, 2]})
ax = axes[0]
for label, r in [("B&H", bh_r), (f"static t={best_t}", static_r),
                  ("HMM no-wc", blend_nowc_r),
                  ("HMM wc-feat", blend_wcfeat_r),
                  ("HMM wc-multi", blend_wcmult_r)]:
    cs = np.where(np.isnan(r), 0.0, np.where(np.isfinite(r), r, 0.0))
    eq = np.exp(np.cumsum(cs))
    eq = np.where(np.isnan(r), np.nan, eq)
    ax.plot(test_idx, eq, lw=0.9, label=label)
ax.set_ylabel("Equity (start=1)")
ax.set_title(f"BTC 1h Risk-Modul mit Thermodynamic Worst-Case (η={eta_calibration})\n"
             f"calibration snippet: {cal_slice.index.min().date()} → {cal_slice.index.max().date()} "
             f"(train fold only)")
ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
for label, r, c in [("B&H", bh_r, "tab:blue"), ("static-soft", static_r, "tab:orange"),
                     ("HMM no-wc", blend_nowc_r, "tab:green"),
                     ("HMM wc-feat", blend_wcfeat_r, "tab:red"),
                     ("HMM wc-multi", blend_wcmult_r, "tab:purple")]:
    rv = pd.Series(r).rolling(24).std(ddof=1).fillna(0) * np.sqrt(PERIODS_PER_YEAR)
    ax.plot(test_idx, rv, lw=0.9, color=c, label=label)
ax.axhline(VOL_TARGET_ANN, color="k", lw=1, ls="--")
ax.set_ylabel("Rolling 24h ann. Vol"); ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)

ax = axes[2]
for label, r, c in [("B&H", bh_r, "tab:blue"), ("static-soft", static_r, "tab:orange"),
                     ("HMM no-wc", blend_nowc_r, "tab:green"),
                     ("HMM wc-feat", blend_wcfeat_r, "tab:red"),
                     ("HMM wc-multi", blend_wcmult_r, "tab:purple")]:
    cs = np.cumsum(pd.Series(r).fillna(0))
    peak = np.maximum.accumulate(cs)
    dd = cs - peak
    ax.fill_between(test_idx, dd, 0, alpha=0.2, color=c, label=label)
ax.set_ylabel("Drawdown (log)"); ax.legend(loc="lower left", fontsize=8, ncol=3); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "wc_equity.png", dpi=130)
plt.close(fig)

# Bootstrap
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, key in zip(axes, ["sharpe", "max_dd"]):
    dist = [boots[n][key] for n in ["B&H", "static-soft", "HMM-no-wc", "HMM-wc-feat", "HMM-wc-multi"]]
    bp = ax.boxplot(dist, tick_labels=["B&H", "static", "no-wc", "wc-feat", "wc-multi"],
                    patch_artist=True, showmeans=True, meanline=True,
                    boxprops=dict(facecolor="lightblue", alpha=0.5))
    for j, c in enumerate(["lightcoral", "lightgreen", "gold"]):
        bp["boxes"][2 + j].set_facecolor(c)
    ax.set_title(key); ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT_DIR / "wc_bootstrap.png", dpi=130)
plt.close(fig)

# Per-window
fig, ax = plt.subplots(figsize=(13, 4))
xs = np.arange(len(win_nowc))
ax.bar(xs - 0.3, [w["test_sharpe"] for w in win_nowc], 0.3, color="tab:green", label="no-wc")
ax.bar(xs,         [w["test_sharpe"] for w in win_wcfeat], 0.3, color="tab:red", label="wc-feat")
ax.bar(xs + 0.3,   [w["test_sharpe"] for w in win_wcmult], 0.3, color="tab:purple", label="wc-multi")
ax.axhline(0, color="k", lw=1)
ax.set_xticks(xs)
ax.set_xticklabels(
    [f"{w['test_start'].date()}\n→ {w['test_end'].date()}" for w in win_nowc], fontsize=7)
ax.set_ylabel("OOS Sharpe"); ax.set_title("Per-Window OOS Sharpe: wc variants")
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT_DIR / "wc_per_window.png", dpi=130)
plt.close(fig)

# Calibration snippet histogram + OOS overlay
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(snippet_logret, bins=80, alpha=0.6, color="tab:blue", density=True, label="calibration snippet")
ax.hist(oos_logret, bins=80, alpha=0.4, color="tab:red", density=True, label="OOS test fold")
ax.set_xlabel("log_return"); ax.set_ylabel("density")
ax.set_title(f"Distribution comparison — snippet dates: {cal_slice.index.min().date()} → {cal_slice.index.max().date()}\n"
             f"KS={ks_stat:.3f}, p={ks_p:.3f}")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "cal_vs_oos_distribution.png", dpi=130)
plt.close(fig)

# ---------- save ----------
out = {
    "eta_calibration": eta_calibration,
    "calibration_snippet": {
        "start": cal_slice.index.min().isoformat(),
        "end": cal_slice.index.max().isoformat(),
        "n_hours": int(len(cal_logret)),
        "mean_loss": meanL,
        "wc_at_eta": v_wc(cal_logret, eta_calibration),
        "ratio_at_eta": v_wc(cal_logret, eta_calibration) / meanL if meanL > 1e-12 else None,
    },
    "post_mortem": {
        "ks_stat": float(ks_stat), "ks_p_value": float(ks_p),
        "snippet_summary": {"mean": float(snippet_logret.mean()), "std": float(snippet_logret.std())},
        "oos_summary":      {"mean": float(oos_logret.mean()),     "std": float(oos_logret.std())},
    },
    "headline": {n: {k: round(v, 6) if isinstance(v, float) else v for k, v in m.items()}
                  for n, m in [("B&H", m_bh), (f"static_soft_t{best_t}", m_static),
                                ("HMM_no_wc", m_nowc),
                                ("HMM_wc_feat", m_wcfeat),
                                ("HMM_wc_multi", m_wcmult)]},
    "p_values": {
        "HMM_wc_multi_vs_no_wc": p_wcmult_vs_nowc,
        "HMM_wc_multi_vs_BH": p_wcmult_vs_bh,
        "HMM_wc_multi_vs_static": p_wcmult_vs_static,
    },
}
(OUT_DIR / "summary.json").write_text(json.dumps(out, indent=2, default=str))
pd.DataFrame(win_nowc).to_csv(OUT_DIR / "per_window_no_wc.csv", index=False)
pd.DataFrame(win_wcfeat).to_csv(OUT_DIR / "per_window_wc_feat.csv", index=False)
pd.DataFrame(win_wcmult).to_csv(OUT_DIR / "per_window_wc_multi.csv", index=False)

print(f"\nSaved to {OUT_DIR}/")
for f in sorted(OUT_DIR.glob("*")):
    print(" -", f, f.stat().st_size, "bytes")
