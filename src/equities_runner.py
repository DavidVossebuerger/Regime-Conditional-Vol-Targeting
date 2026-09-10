"""Equity pipeline (daily bars via London Strategic Edge).

Mirrors multi_asset_runner.py but uses DAILY bars for equities (US stocks
trade ~6.5h/day, so daily is the right granularity). Reuses the same
HMM wc-feat strategy.

Differences vs crypto runner:
- periods_per_year = 252 (trading days)
- roll_window = 20 (days) for daily-equivalent realized vol
- walk-forward: train_min=504 (~2y), test_size=126 (~6mo), step=126
- bootstrap block_size = 60 (trading days ≈ 3 months)
"""
from __future__ import annotations
import argparse
import json
import warnings
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestRegressor

from data_io.lse_loader import load_lse_daily

warnings.filterwarnings("ignore")

# Daily-equity config
PERIODS_PER_YEAR = 252
ROLL = 20
LAGS = 5
SEED = 42
K_HMM = 3
TARGET_GRID = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
WF_TRAIN_MIN = 504
WF_TEST_SIZE = 126
WF_STEP = 126
WC_WINDOW = 60                  # 60 trading days ≈ 3 months
EPS = 1e-3
ETA_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
SPLIT = 0.8
CAL_SNIPPET_DAYS = 90
CAL_SEED = 7
N_BOOT = 1000
BLOCK_SIZE = 60
TC_PER_SIDE = 0.0002              # 2 bps / side (retail equity)
RF_PARAMS = dict(n_estimators=125, max_depth=24, max_features="sqrt",
                 min_samples_leaf=12, n_jobs=-1, random_state=SEED)

# Default small-cap basket — US small/mid caps with reasonable history via LSE
EQUITY_UNIVERSE = [
    "RIOT",   # Riot Platforms — crypto miner, vol-heavy
    "MARA",   # Marathon Digital — crypto miner
    "DKNG",   # DraftKings — gaming/sports betting
    "LCID",   # Lucid Motors — EV
    "RIVN",   # Rivian — EV
    "PLTR",   # Palantir — software
    "RKT",    # Rocket Companies — mortgage fintech
    "OPEN",   # Opendoor — iBuyer
    "SOFI",   # SoFi — fintech
]


def build_features(close: pd.Series, window: int = ROLL, lags: int = LAGS) -> pd.DataFrame:
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


def soft_position(target_vol, pred_vals):
    return np.clip(target_vol / np.maximum(pred_vals, EPS), 0.0, 1.0)


def metrics(r, pos):
    if len(r) < 50 or r.std(ddof=1) < 1e-12:
        return {"ann_ret": 0.0, "ann_vol": 0.0, "sharpe": np.nan,
                "max_dd": 0.0, "pct_within": float("nan")}
    sd = float(r.std(ddof=1))
    ann_r = r.mean() * PERIODS_PER_YEAR
    ann_v = sd * np.sqrt(PERIODS_PER_YEAR)
    eq = np.cumsum(r); peak = np.maximum.accumulate(eq)
    max_dd = float(-(eq - peak).min())
    roll_v = pd.Series(r).rolling(20).std(ddof=1).dropna() * np.sqrt(PERIODS_PER_YEAR)
    pct_within = float((roll_v <= 0.40).mean()) if len(roll_v) else float("nan")
    return dict(ann_ret=ann_r, ann_vol=ann_v, sharpe=ann_r / ann_v,
                max_dd=max_dd, pct_within=pct_within)


def sharpe_rank(r):
    if len(r) < 30 or r.std(ddof=1) < 1e-12: return -np.inf
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


def run_one_symbol(symbol: str, OUT_DIR: Path):
    print(f"\n{'='*60}\n  {symbol}\n{'='*60}")
    close = load_lse_daily(symbol)
    if close is None or len(close) < 800:
        print(f"  insufficient data ({len(close) if close is not None else 0} bars); skipping")
        return None

    log_ret_full = np.log(close / close.shift(1)).dropna()
    feat = build_features(close)
    train_n = int(len(feat) * SPLIT)
    train = feat.iloc[:train_n]
    test = feat.iloc[train_n:]
    test_idx = test.index
    N_TEST = len(test_idx)
    print(f"  train={len(train)}, test={N_TEST} bars  ({close.index.min().date()} → {close.index.max().date()})")

    X_train = train.drop(columns=["target"]); y_train = train["target"]
    X_test = test.drop(columns=["target"]);   y_test = test["target"]

    rf = RandomForestRegressor(**RF_PARAMS).fit(X_train, y_train)
    pred = pd.Series(rf.predict(X_test), index=test_idx, name="pred")
    pred_arr = pred.values
    log_ret_arr = log_ret_full.reindex(test_idx).fillna(0).values

    # Calibration
    rng = np.random.default_rng(CAL_SEED)
    max_start = len(train) - CAL_SNIPPET_DAYS - 1
    if max_start <= 0:
        chosen_eta = ETA_GRID[-1]
    else:
        cal_start = int(rng.integers(0, max_start))
        cal_logret = train["log_return"].iloc[cal_start:cal_start + CAL_SNIPPET_DAYS].values
        meanL = float(np.mean(cal_logret))
        chosen_eta = ETA_GRID[-1]
        for eta in ETA_GRID:
            wc = v_wc(cal_logret, eta)
            drag = abs(wc - meanL) / max(abs(meanL), 1e-6)
            if drag >= 0.5:
                chosen_eta = eta
                break
    print(f"  η_calibration={chosen_eta}")

    wc_full = wc_series(log_ret_arr, chosen_eta, WC_WINDOW)
    wc_test = pd.Series(wc_full, index=test_idx).shift(1).fillna(0.0).values

    rv_realized = (log_ret_full * np.sqrt(PERIODS_PER_YEAR)).reindex(test_idx).fillna(0).values
    rv_lag1 = pd.Series(rv_realized).shift(1).fillna(0.0).values
    vol_zscore = pd.Series(rv_lag1).rolling(60).apply(
        lambda s: (s.iloc[-1] - s.mean()) / (s.std() + 1e-12), raw=False
    ).fillna(0.0).values
    vol_of_vol = pd.Series(rv_lag1).rolling(20).std().fillna(0.0).values
    vol_return = np.concatenate([[0.0], np.diff(rv_lag1)])

    hmm_X = np.column_stack([vol_zscore, vol_of_vol, vol_return, wc_test])
    hmm_X = np.nan_to_num(hmm_X, nan=0.0, posinf=0.0, neginf=0.0)
    mu = hmm_X.mean(axis=0); sd = hmm_X.std(axis=0) + 1e-12
    hmm_X_std = (hmm_X - mu) / sd
    first_valid = 60
    hmm_X_eff = hmm_X_std[first_valid:]
    eff_idx = test_idx[first_valid:]

    # Adaptive WF sizing: for short histories, scale down WF_TRAIN_MIN so we
    # can fit at least 1 walk-forward window.
    eff_len = len(eff_idx)
    min_wf = min(WF_TRAIN_MIN, max(60, eff_len // 3))
    test_size = min(WF_TEST_SIZE, max(40, eff_len // 6))
    step = test_size  # non-overlapping windows for short histories

    windows = []
    ww = 0
    while True:
        te = ww + min_wf
        if te + 30 >= eff_len: break
        fs_eff = te
        fe_eff = min(fs_eff + test_size, eff_len)
        if fe_eff - fs_eff < 40: break
        windows.append((ww, te, first_valid + fs_eff, first_valid + fe_eff, fe_eff - fs_eff))
        ww += step
        if fe_eff >= eff_len: break

    print(f"  walk-forward windows: {len(windows)} (adaptive: train_min={min_wf}, test_size={test_size})")
    if len(windows) < 1:
        return None

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
    for ww in windows: covered[ww[2]:ww[3]] = True

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

    boots_bh = block_bootstrap(bh_r[covered], N_BOOT, BLOCK_SIZE, SEED)
    boots_blend = block_bootstrap(blend_r[covered], N_BOOT, BLOCK_SIZE, SEED + 1)
    p_beats = float(np.mean(np.asarray(boots_blend) >= np.asarray(boots_bh)))

    df_yearly = []
    for yr in sorted(set(test_idx.year)):
        mask = (test_idx.year == yr) & covered
        if mask.sum() < 30: continue
        m_b = metrics(bh_r[mask], np.ones(mask.sum()))
        m_h = metrics(blend_r[mask], blend_pos_full[mask])
        df_yearly.append({
            "year": int(yr), "n_days": int(mask.sum()),
            "bh_sharpe": m_b["sharpe"], "hmm_sharpe": m_h["sharpe"],
            "hmm_max_dd": m_h["max_dd"],
        })

    out = {
        "symbol": symbol, "n_test_bars": int(N_TEST),
        "covered_bars": int(covered.sum()), "n_windows": len(windows),
        "cal_eta": chosen_eta,
        "headline_bh": m_bh, "headline_hmm": m_blend,
        "p_hmm_beats_bh": p_beats,
        "yearly": df_yearly,
    }
    print(f"  bh_sharpe={m_bh['sharpe']:.3f}  hmm_sharpe={m_blend['sharpe']:.3f}  "
          f"max_dd_bh={m_bh['max_dd']:.3f}  max_dd_hmm={m_blend['max_dd']:.3f}  "
          f"p_beats={p_beats:.3f}")

    asset_dir = OUT_DIR / symbol
    asset_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(win_results).to_csv(asset_dir / "per_window.csv", index=False)
    pd.DataFrame(df_yearly).to_csv(asset_dir / "yearly.csv", index=False)
    (asset_dir / "summary.json").write_text(json.dumps(out, indent=2, default=str))

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                              gridspec_kw={"height_ratios": [3, 2, 2]})
    ax = axes[0]
    for label, r, c in [("B&H", bh_r, "tab:blue"), ("HMM wc-feat", blend_r, "tab:red")]:
        cs = np.where(np.isnan(r), 0.0, np.where(np.isfinite(r), r, 0.0))
        eq = np.exp(np.cumsum(cs))
        eq = np.where(np.isnan(r), np.nan, eq)
        ax.plot(test_idx, eq, lw=0.9, color=c, label=label)
    ax.set_ylabel("Equity (start=1)")
    ax.set_title(f"{symbol} (equity, daily) HMM wc-feat vs B&H (η={chosen_eta})\n"
                 f"sharpe_bh={m_bh['sharpe']:.2f}, hmm={m_blend['sharpe']:.2f}, "
                 f"P(hmm≥bh)={p_beats:.2f}")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    for label, r, c in [("B&H", bh_r, "tab:blue"), ("HMM", blend_r, "tab:red")]:
        rv = pd.Series(r).rolling(20).std(ddof=1).fillna(0) * np.sqrt(PERIODS_PER_YEAR)
        ax.plot(test_idx, rv, lw=0.8, color=c, label=label)
    ax.axhline(0.40, color="k", lw=1, ls="--", label="Vol-Target 40%")
    ax.set_ylabel("Rolling 20d ann. Vol"); ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    for label, r, c in [("B&H", bh_r, "tab:blue"), ("HMM", blend_r, "tab:red")]:
        cs = np.cumsum(pd.Series(r).fillna(0))
        peak = np.maximum.accumulate(cs)
        dd = cs - peak
        ax.fill_between(test_idx, dd, 0, alpha=0.3, color=c, label=label)
    ax.set_ylabel("Drawdown (log)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(asset_dir / "equity.png", dpi=110)
    plt.close(fig)

    return out


def main():
    import time as _time
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=",".join(EQUITY_UNIVERSE))
    ap.add_argument("--output-dir", default="outputs/equities")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--from-csv", default=None,
                    help="Load symbols from a CSV (column 'ticker'). Used for Russell 2000 batches.")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N symbols (debugging)")
    args = ap.parse_args()

    if args.from_csv:
        import csv as _csv
        with open(args.from_csv) as f:
            r = _csv.DictReader(f)
            symbols = [row["ticker"].strip() for row in r if row.get("ticker")]
        symbols = list(dict.fromkeys(symbols))
        print(f"Loaded {len(symbols)} symbols from {args.from_csv}")
    else:
        symbols = list(dict.fromkeys(s.strip().upper() for s in args.assets.split(",") if s.strip()))
    if args.limit:
        symbols = symbols[:args.limit]

    if args.no_cache:
        from data_io.lse_loader import clear_cache
        clear_cache()

    OUT_DIR = Path(args.output_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Equities risk-management pipeline ===")
    print(f"  symbols: {len(symbols)} ({', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''})")
    print(f"  output:  {OUT_DIR}")
    print(f"  periods/yr: {PERIODS_PER_YEAR} (trading days)")

    rows = []
    t_start = _time.time()
    n_total = len(symbols)
    for i, sym in enumerate(symbols, 1):
        t0 = _time.time()
        try:
            r = run_one_symbol(sym, OUT_DIR)
        except Exception as e:
            print(f"  ERROR {sym}: {e}")
            r = None
        dt = _time.time() - t0
        if r is not None:
            rows.append(r)
        # Progress + ETA
        done = len(rows)
        skipped = i - done
        elapsed = _time.time() - t_start
        avg = elapsed / i if i else 0
        eta = avg * (n_total - i)
        status = "✓" if r is not None else "✗"
        delta_s = r["headline_hmm"]["sharpe"] - r["headline_bh"]["sharpe"] if r else float("nan")
        print(f"  [{i:>3d}/{n_total}] {status} {sym:<8s} "
              f"{dt:>5.1f}s  Δ={delta_s:+.3f}  "
              f"avg={avg:>5.1f}s  eta={_time.strftime('%H:%M:%S', _time.gmtime(eta))}  "
              f"({done} ok, {skipped} skipped)")

    if rows:
        df = pd.DataFrame([{
            "symbol": r["symbol"], "n_test_bars": r["n_test_bars"],
            "n_windows": r["n_windows"], "cal_eta": r["cal_eta"],
            "bh_sharpe": r["headline_bh"]["sharpe"],
            "hmm_sharpe": r["headline_hmm"]["sharpe"],
            "sharpe_delta": r["headline_hmm"]["sharpe"] - r["headline_bh"]["sharpe"],
            "bh_max_dd": r["headline_bh"]["max_dd"],
            "hmm_max_dd": r["headline_hmm"]["max_dd"],
            "p_hmm_beats_bh": r["p_hmm_beats_bh"],
        } for r in rows]).sort_values("sharpe_delta", ascending=False)
        print("\n" + "=" * 60)
        print("EQUITIES SUMMARY (sorted by Δ Sharpe)")
        print("=" * 60)
        print(df.round(3).to_string(index=False))
        df.to_csv(OUT_DIR / "summary.csv", index=False)

        # Cross-asset PNG
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax = axes[0]
        xs = np.arange(len(df)); w = 0.4
        ax.bar(xs - w/2, df["bh_sharpe"], w, color="tab:blue", label="B&H")
        ax.bar(xs + w/2, df["hmm_sharpe"], w, color="tab:red", label="HMM wc-feat")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(xs); ax.set_xticklabels(df["symbol"], rotation=45, ha="right")
        ax.set_ylabel("Sharpe"); ax.set_title("Sharpe — US Small-Cap Equities (daily)")
        ax.legend(); ax.grid(alpha=0.3, axis="y")

        ax = axes[1]
        ax.bar(xs, df["sharpe_delta"],
               color=["tab:green" if d > 0 else "tab:red" for d in df["sharpe_delta"]])
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(xs); ax.set_xticklabels(df["symbol"], rotation=45, ha="right")
        ax.set_ylabel("Sharpe Δ (HMM - B&H)")
        ax.set_title("Where HMM wc-feat beats B&H on equities")
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "cross_asset.png", dpi=110)
        plt.close(fig)
        print(f"\nSaved to {OUT_DIR}/")
        for f in sorted(OUT_DIR.glob("*")):
            print(" -", f, "(dir)" if f.is_dir() else f"{f.stat().st_size} bytes")


if __name__ == "__main__":
    main()
