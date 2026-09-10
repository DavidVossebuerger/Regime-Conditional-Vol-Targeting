"""Multi-asset risk-modul validation: HMM wc-feat vs B&H across all available crypto.

For each asset:
  - load 1h bars (parquet for BTC/ETH, CSV for alts)
  - 80/20 train/test split
  - 90-day random calibration snippet from train fold only
  - Walk-forward HMM wc-feat (24 windows default)
  - Bootstrap CI (N=1000 for speed across many assets)
  - Per-asset PNGs + summary

Outputs:
  results_smoke/multi_asset/
    per_asset_summary.csv
    cross_asset.png (Sharpe / Max DD / vol-target hit per asset)
    {asset}/equity.png
    {asset}/per_window.png
    {asset}/yearly.csv
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

warnings.filterwarnings("ignore")

DATA_DIR = Path("/home/davidv/Dokumente/Risikooptimierung/data")
OUT_DIR = Path("results_smoke/multi_asset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline config (consistent with the BTC script)
TC_PER_SIDE = 0.0002
PERIODS_PER_YEAR = 8760
ROLL = 24
LAGS = 5
SEED = 42
K_HMM = 3
TARGET_GRID = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
WF_TRAIN_MIN = 4320       # 180 days
WF_TEST_SIZE = 2160       # 90 days
WF_STEP = 2160
WC_WINDOW = 168
EPS = 1e-3
ETA_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
SPLIT = 0.8
CAL_SNIPPET_HOURS = 90 * 24
CAL_SEED = 7
N_BOOT = 1000
BLOCK_SIZE = 168

# RF — fixed params (from BTC Optuna best)
RF_PARAMS = dict(n_estimators=125, max_depth=24, max_features="sqrt",
                 min_samples_leaf=12, n_jobs=-1, random_state=SEED)

# ---------- loaders ----------
def load_parquet_1h(pattern_prefix: str) -> pd.Series:
    files = sorted(DATA_DIR.glob(f"{pattern_prefix}*1m*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values("ts").drop_duplicates("ts")
    df["Date"] = pd.to_datetime(df["ts"])
    df = df.set_index("Date")[["close"]].astype(float)
    h1 = df.resample("1h").last().dropna()
    return h1["close"]


def load_csv_1h(filename: str) -> pd.Series:
    df = pd.read_csv(DATA_DIR / filename)
    df.columns = [c.lower() for c in df.columns]
    # CSV timestamps may be ISO strings ("2018-04-17 04:00:00+00") — let pandas auto-parse.
    ts = df["timestamp"] if "timestamp" in df.columns else df.iloc[:, 0]
    df["Date"] = pd.to_datetime(ts, errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    return df["close"].astype(float)


# Define all assets with their loader
ASSETS = [
    ("BTC", "parquet:1m:crypto_BTC_USD"),
    ("ETH", "parquet:1m:crypto_ETH_USD"),
    ("ADA",  "csv:ada_usd_1h.csv"),
    ("BNB",  "csv:bnb_usd_1h.csv"),
    ("DOGE", "csv:doge_usd_1h.csv"),
    ("FDUSD","csv:fdusd_usd_1h.csv"),
    ("LINK", "csv:link_usd_1h.csv"),
    ("LTC",  "csv:ltc_usd_1h.csv"),
    ("SHIB", "csv:shib_usd_1h.csv"),
    ("SOL",  "csv:sol_usd_1h.csv"),
    ("USDC", "csv:usdc_usd_1h.csv"),
    ("XRP",  "csv:xrp_usd_1h.csv"),
]


def load_asset(spec) -> pd.Series | None:
    if spec.startswith("parquet:"):
        parts = spec.split(":")
        prefix = parts[2]
        try:
            return load_parquet_1h(prefix)
        except Exception as e:
            print(f"  parquet load failed for {spec}: {e}")
            return None
    elif spec.startswith("csv:"):
        try:
            return load_csv_1h(spec.split(":", 1)[1])
        except Exception as e:
            print(f"  csv load failed for {spec}: {e}")
            return None
    return None


# ---------- feature engineering ----------
def build_features(close, window=ROLL, lags=LAGS):
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
    pct_within = float((roll_v <= 0.6).mean()) if len(roll_v) else float("nan")
    return dict(ann_ret=ann_r, ann_vol=ann_v, sharpe=ann_r / ann_v,
                max_dd=max_dd, pct_within=pct_within)


def sharpe_rank(r):
    if len(r) < 100 or r.std(ddof=1) < 1e-12: return -np.inf
    return float(r.mean() / r.std(ddof=1))


def v_wc(logret, eta):
    z = np.log(np.mean(np.exp(eta * logret)))
    return float(z / eta)


def wc_series(logret, eta, window):
    out = np.full(len(logret), np.nan)
    for i in range(window, len(logret)):
        out[i] = v_wc(logret[i - window:i], eta)
    return out


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


def block_bootstrap(returns, n_boot, block, seed):
    n = len(returns); n_blocks = (n + block - 1) // block
    starts = np.arange(0, n, block)
    rng = np.random.default_rng(seed)
    sharpes = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(starts), size=n_blocks)
        sample = np.concatenate([returns[s:min(s + block, n)] for s in starts[chosen]])[:n]
        sd = float(np.std(sample, ddof=1))
        if sd < 1e-12: continue
        sharpes.append((sample.mean() * PERIODS_PER_YEAR) / (sd * np.sqrt(PERIODS_PER_YEAR)))
    return sharpes


# ---------- per-asset pipeline ----------
def run_one_asset(name: str, close: pd.Series):
    print(f"\n{'='*70}\n  {name}\n{'='*70}")
    log_ret_full = np.log(close / close.shift(1)).dropna()
    feat = build_features(close)
    train_n = int(len(feat) * SPLIT)
    train = feat.iloc[:train_n]
    test = feat.iloc[train_n:]
    test_idx = test.index
    N_TEST = len(test_idx)
    if N_TEST < WF_TEST_SIZE * 2:
        print(f"  too few test bars ({N_TEST}); skipping")
        return None

    X_train = train.drop(columns=["target"]); y_train = train["target"]
    X_test = test.drop(columns=["target"]);   y_test = test["target"]

    rf = RandomForestRegressor(**RF_PARAMS).fit(X_train, y_train)
    pred = pd.Series(rf.predict(X_test), index=test_idx, name="pred")
    pred_arr = pred.values
    log_ret_arr = log_ret_full.reindex(test_idx).fillna(0).values

    # Calibration
    rng = np.random.default_rng(CAL_SEED)
    max_start = len(train) - CAL_SNIPPET_HOURS - 1
    cal_start = int(rng.integers(0, max_start))
    cal_logret = train["log_return"].iloc[cal_start:cal_start + CAL_SNIPPET_HOURS].values
    meanL = float(np.mean(cal_logret))
    chosen_eta = ETA_GRID[-1]
    for eta in ETA_GRID:
        wc = v_wc(cal_logret, eta)
        drag = abs(wc - meanL) / max(abs(meanL), 1e-6)
        if drag >= 0.5:
            chosen_eta = eta
            break
    print(f"  train={len(train)}, test={N_TEST} bars  cal_start=iloc[{cal_start}]  η={chosen_eta}")

    wc_full = wc_series(log_ret_arr, chosen_eta, WC_WINDOW)
    wc_test = pd.Series(wc_full, index=test_idx).shift(1).fillna(0.0).values

    rv_realized = (log_ret_full * np.sqrt(PERIODS_PER_YEAR)).reindex(test_idx).fillna(0).values
    rv_lag1 = pd.Series(rv_realized).shift(1).fillna(0.0).values
    vol_zscore = pd.Series(rv_lag1).rolling(168).apply(
        lambda s: (s.iloc[-1] - s.mean()) / (s.std() + 1e-12), raw=False
    ).fillna(0.0).values
    vol_of_vol = pd.Series(rv_lag1).rolling(72).std().fillna(0.0).values
    vol_return = np.concatenate([[0.0], np.diff(rv_lag1)])

    cols = [vol_zscore, vol_of_vol, vol_return, wc_test]
    hmm_X = np.column_stack(cols)
    hmm_X = np.nan_to_num(hmm_X, nan=0.0, posinf=0.0, neginf=0.0)
    mu = hmm_X.mean(axis=0); sd = hmm_X.std(axis=0) + 1e-12
    hmm_X_std = ((hmm_X - mu) / sd)
    first_valid = 168
    hmm_X_eff = hmm_X_std[first_valid:]
    eff_idx = test_idx[first_valid:]

    # Walk-forward
    windows = []
    w = 0
    while True:
        te = w + WF_TRAIN_MIN
        if te + 168 >= len(eff_idx): break
        fs_eff = te
        fe_eff = min(fs_eff + WF_TEST_SIZE, len(eff_idx))
        if fe_eff - fs_eff < 720: break
        windows.append((w, te, first_valid + fs_eff, first_valid + fe_eff, fe_eff - fs_eff))
        w += WF_STEP
        if fe_eff >= len(eff_idx): break

    print(f"  walk-forward windows: {len(windows)}")
    if len(windows) < 3:
        return None

    blend_pos_full = np.full(N_TEST, np.nan)
    win_results = []
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

        r_win = pos_window * log_ret_arr[test_orig_idx]
        tc_win = np.abs(np.diff(pos_window, prepend=pos_window[0])) * TC_PER_SIDE
        r_net = r_win - tc_win
        m = metrics(r_net, pos_window)
        win_results.append({
            "window": w_idx, "test_start": test_idx[fs], "test_end": test_idx[fe - 1],
            "test_sharpe": m["sharpe"], "test_max_dd": m["max_dd"],
            "pct_invested": float(pos_window.mean()),
        })

    covered = np.zeros(N_TEST, dtype=bool)
    for ww in windows:
        covered[ww[2]:ww[3]] = True

    blend_r = np.full(N_TEST, np.nan)
    for ww in windows:
        fs, fe = ww[2], ww[3]
        pos = blend_pos_full[fs:fe]
        r = pos * log_ret_arr[fs:fe]
        tc = np.abs(np.diff(pos, prepend=pos[0])) * TC_PER_SIDE
        blend_r[fs:fe] = r - tc

    bh_r = log_ret_arr.copy()
    m_bh = metrics(bh_r[covered], np.ones(covered.sum()))
    m_blend = metrics(blend_r[covered], blend_pos_full[covered])

    # bootstrap
    boots_bh = block_bootstrap(bh_r[covered], N_BOOT, BLOCK_SIZE, SEED)
    boots_blend = block_bootstrap(blend_r[covered], N_BOOT, BLOCK_SIZE, SEED + 1)
    p_beats = float(np.mean(np.asarray(boots_blend) >= np.asarray(boots_bh)))

    # yearly breakdown
    df_yearly = []
    for yr in sorted(set(test_idx.year)):
        mask = (test_idx.year == yr) & covered
        if mask.sum() < 30:
            continue
        m_b = metrics(bh_r[mask], np.ones(mask.sum()))
        m_h = metrics(blend_r[mask], blend_pos_full[mask])
        df_yearly.append({
            "year": yr, "n_hours": int(mask.sum()),
            "bh_sharpe": m_b["sharpe"], "hmm_sharpe": m_h["sharpe"],
            "bh_ret": m_b["ann_ret"], "hmm_ret": m_h["ann_ret"],
            "hmm_max_dd": m_h["max_dd"], "hmm_pct_inv": float(blend_pos_full[mask].mean()),
        })
    df_yearly = pd.DataFrame(df_yearly)

    out = {
        "asset": name,
        "n_test_bars": int(N_TEST),
        "covered_bars": int(covered.sum()),
        "n_windows": len(windows),
        "cal_eta": chosen_eta,
        "headline_bh": m_bh,
        "headline_hmm": m_blend,
        "sharpe_bh_bootstrap_mean": float(np.mean(boots_bh)),
        "sharpe_hmm_bootstrap_mean": float(np.mean(boots_blend)),
        "p_hmm_beats_bh": p_beats,
        "yearly": df_yearly.to_dict(orient="records"),
        "windows_meta": [{"test_start": w["test_start"].date().isoformat(),
                          "test_end": w["test_end"].date().isoformat(),
                          "test_sharpe": float(w["test_sharpe"]) if w["test_sharpe"] == w["test_sharpe"] else None,
                          "pct_invested": float(w["pct_invested"])} for w in win_results],
    }
    print(f"  bh_sharpe={m_bh['sharpe']:.3f}  hmm_sharpe={m_blend['sharpe']:.3f}  "
          f"max_dd_bh={m_bh['max_dd']:.3f}  max_dd_hmm={m_blend['max_dd']:.3f}  "
          f"p_beats={p_beats:.3f}")

    # per-asset outputs
    asset_dir = OUT_DIR / name
    asset_dir.mkdir(exist_ok=True)
    pd.DataFrame(win_results).to_csv(asset_dir / "per_window.csv", index=False)
    df_yearly.to_csv(asset_dir / "yearly.csv", index=False)

    # PNG: equity / vol / DD
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                              gridspec_kw={"height_ratios": [3, 2, 2]})
    ax = axes[0]
    for label, r, c in [("B&H", bh_r, "tab:blue"),
                          ("HMM wc-feat", blend_r, "tab:red")]:
        cs = np.where(np.isnan(r), 0.0, np.where(np.isfinite(r), r, 0.0))
        eq = np.exp(np.cumsum(cs))
        eq = np.where(np.isnan(r), np.nan, eq)
        ax.plot(test_idx, eq, lw=0.9, color=c, label=label)
    ax.set_ylabel("Equity (start=1)")
    ax.set_title(f"{name} 1h Risk-Modul: B&H vs HMM wc-feat (η={chosen_eta})\n"
                 f"sharpe_bh={m_bh['sharpe']:.2f}, hmm={m_blend['sharpe']:.2f}, "
                 f"P(hmm≥bh)={p_beats:.2f}")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    for label, r, c in [("B&H", bh_r, "tab:blue"), ("HMM wc-feat", blend_r, "tab:red")]:
        rv = pd.Series(r).rolling(24).std(ddof=1).fillna(0) * np.sqrt(PERIODS_PER_YEAR)
        ax.plot(test_idx, rv, lw=0.8, color=c, label=label)
    ax.axhline(0.6, color="k", lw=1, ls="--", label="Vol-Target 60%")
    ax.set_ylabel("Rolling 24h ann. Vol"); ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    for label, r, c in [("B&H", bh_r, "tab:blue"), ("HMM wc-feat", blend_r, "tab:red")]:
        cs = np.cumsum(pd.Series(r).fillna(0))
        peak = np.maximum.accumulate(cs)
        dd = cs - peak
        ax.fill_between(test_idx, dd, 0, alpha=0.3, color=c, label=label)
    ax.set_ylabel("Drawdown (log)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(asset_dir / "equity.png", dpi=110)
    plt.close(fig)

    return out


# ---------- run all ----------
import argparse as _argparse
_parser = _argparse.ArgumentParser()
_parser.add_argument("--assets", default="all",
                    help="Comma-separated asset names or 'all'")
_parser.add_argument("--output-dir", default="results_smoke/multi_asset",
                    help="Output directory")
_args, _ = _parser.parse_known_args()

if _args.assets.lower() == "all":
    _selected = ASSETS
else:
    _selected_set = set(a.strip().upper() for a in _args.assets.split(","))
    _selected = [(n, s) for n, s in ASSETS if n.upper() in _selected_set]

OUT_DIR = Path(_args.output_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

all_results = []
for name, spec in _selected:
    close = load_asset(spec)
    if close is None or len(close) < 5000:
        print(f"\n{name}: insufficient data, skipping")
        continue
    print(f"\n{name}: {len(close)} hourly bars, {close.index.min().date()} → {close.index.max().date()}")
    try:
        out = run_one_asset(name, close)
    except Exception as e:
        print(f"  ERROR: {e}")
        out = None
    if out is not None:
        all_results.append(out)
        (OUT_DIR / name / "summary.json").write_text(json.dumps(out, indent=2, default=str))

# ---------- cross-asset summary ----------
print("\n" + "=" * 70)
print("CROSS-ASSET SUMMARY")
print("=" * 70)

rows = []
for o in all_results:
    hmm = o["headline_hmm"]; bh = o["headline_bh"]
    rows.append({
        "asset": o["asset"],
        "n_test_bars": o["n_test_bars"],
        "n_windows": o["n_windows"],
        "cal_eta": o["cal_eta"],
        "bh_sharpe": bh["sharpe"], "hmm_sharpe": hmm["sharpe"],
        "sharpe_delta": hmm["sharpe"] - bh["sharpe"],
        "bh_max_dd": bh["max_dd"], "hmm_max_dd": hmm["max_dd"],
        "dd_improvement": bh["max_dd"] - hmm["max_dd"],   # positive = HMM less negative
        "bh_vol": bh["ann_vol"], "hmm_vol": hmm["ann_vol"],
        "hmm_pct_invested": float(np.mean([w["pct_invested"] for w in o["windows_meta"]])),
        "p_hmm_beats_bh": o["p_hmm_beats_bh"],
    })
summary = pd.DataFrame(rows).sort_values("sharpe_delta", ascending=False)
print(summary.round(3).to_string(index=False))
summary.to_csv(OUT_DIR / "per_asset_summary.csv", index=False)

# cross-asset PNG
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
ax = axes[0, 0]
xs = np.arange(len(summary))
w = 0.4
ax.bar(xs - w/2, summary["bh_sharpe"], w, color="tab:blue", label="B&H")
ax.bar(xs + w/2, summary["hmm_sharpe"], w, color="tab:red", label="HMM wc-feat")
ax.axhline(0, color="k", lw=0.5)
ax.set_xticks(xs); ax.set_xticklabels(summary["asset"], rotation=45, ha="right")
ax.set_ylabel("Sharpe"); ax.set_title("Sharpe per asset")
ax.legend(); ax.grid(alpha=0.3, axis="y")

ax = axes[0, 1]
ax.bar(xs - w/2, summary["bh_max_dd"], w, color="tab:blue", label="B&H")
ax.bar(xs + w/2, summary["hmm_max_dd"], w, color="tab:red", label="HMM wc-feat")
ax.set_xticks(xs); ax.set_xticklabels(summary["asset"], rotation=45, ha="right")
ax.set_ylabel("Max Drawdown (log, less negative = better)")
ax.set_title("Max DD per asset"); ax.legend(); ax.grid(alpha=0.3, axis="y")

ax = axes[1, 0]
colors = ["tab:green" if d > 0 else "tab:red" for d in summary["sharpe_delta"]]
ax.bar(xs, summary["sharpe_delta"], color=colors)
ax.axhline(0, color="k", lw=1)
ax.set_xticks(xs); ax.set_xticklabels(summary["asset"], rotation=45, ha="right")
ax.set_ylabel("Sharpe Delta (HMM - B&H)")
ax.set_title("Where does HMM beat B&H? (green = yes)")
ax.grid(alpha=0.3, axis="y")

ax = axes[1, 1]
ax.bar(xs, summary["p_hmm_beats_bh"], color=["tab:green" if p > 0.5 else "tab:red" for p in summary["p_hmm_beats_bh"]])
ax.axhline(0.5, color="k", lw=1, ls="--", label="50/50")
ax.set_xticks(xs); ax.set_xticklabels(summary["asset"], rotation=45, ha="right")
ax.set_ylabel("P(HMM Sharpe ≥ B&H)")
ax.set_title("Bootstrap probability HMM ≥ B&H")
ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(OUT_DIR / "cross_asset.png", dpi=120)
plt.close(fig)

# yearly heatmap per asset
print("\nYearly breakdown (ann. return, HMM / B&H):")
for o in all_results:
    print(f"\n{o['asset']}:")
    df = pd.DataFrame(o["yearly"])
    if len(df):
        print(df[["year", "n_hours", "bh_sharpe", "hmm_sharpe", "hmm_pct_inv"]].round(3).to_string(index=False))

print(f"\nSaved to {OUT_DIR}/")
for f in sorted(OUT_DIR.glob("*")):
    print(" -", f, "(dir)" if f.is_dir() else f"{f.stat().st_size} bytes")
