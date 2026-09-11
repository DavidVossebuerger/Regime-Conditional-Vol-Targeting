"""Regime-dependent ΔSharpe decomposition.

Question: Is RCVT's edge uniform across HMM regimes, or concentrated
in one (or two) regimes? Squiggle's Discord intuition: "kinda sketchy how
well it did on SOL, I'd imagine it's regime dependent."

If concentrated → strategy is effectively bull/bear timing dressed as
regime detection. If uniform → real regime-conditional alpha.

Reads per-bar CSVs produced by `multi_asset_runner.py --dump-bars` and
reports per-regime per-asset ΔSharpe + the share of total edge that
comes from each regime.

Usage:
    # 1) Produce per-bar CSVs (one-off, slow):
    python src/multi_asset_runner.py --assets SOL,BTC,ETH --dump-bars --output-dir outputs/multi_asset
    # 2) Analyze:
    python tests/regime_decomposition.py
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Same annualization as the runners (hourly crypto 24/7)
PERIODS_PER_YEAR = 8760


def per_regime_stats(bars: pd.DataFrame) -> pd.DataFrame:
    """For each state k, compute Sharpe_bh, Sharpe_rcvt, ΔSharpe on the bars
    where argmax posterior = k (only `covered` bars).
    """
    df = bars[bars["covered"] & (bars["state"] >= 0)].copy()
    rows = []
    for k in sorted(df["state"].unique()):
        sub = df[df["state"] == k]
        bh = sub["bh_logret"].dropna().values
        rcvt = sub["rcvt_logret"].dropna().values
        if len(bh) < 50 or bh.std(ddof=1) < 1e-12 or rcvt.std(ddof=1) < 1e-12:
            rows.append(dict(state=int(k), n_bars=int(len(sub)),
                            pct_of_test=float(len(sub)) / max(len(df), 1),
                            bh_sharpe=float("nan"), rcvt_sharpe=float("nan"),
                            delta_sharpe=float("nan")))
            continue
        bh_sh = bh.mean() / bh.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)
        rcvt_sh = rcvt.mean() / rcvt.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)
        rows.append(dict(state=int(k), n_bars=int(len(sub)),
                        pct_of_test=float(len(sub)) / max(len(df), 1),
                        bh_sharpe=float(bh_sh), rcvt_sharpe=float(rcvt_sh),
                        delta_sharpe=float(rcvt_sh - bh_sh)))
    return pd.DataFrame(rows)


def concentration(per_regime: pd.DataFrame) -> float:
    """Share of total positive edge coming from the strongest single regime.

    Concentration near 1.0 ⇢ edge is regime-specific.
    Concentration < 0.6 ⇢ edge is broadly distributed.
    Returns NaN if no positive edge exists.
    """
    pos = per_regime["delta_sharpe"].clip(lower=0).sum()
    if pos <= 0:
        return float("nan")
    return float(per_regime["delta_sharpe"].clip(lower=0).max() / pos)


def soft_weighted(per_regime: pd.DataFrame, bars: pd.DataFrame) -> dict:
    """Decompose edge using posterior-weighted regime attribution.
    For each bar, contribution to regime k = rcvt_logret · p_k - bh_logret · p_k.
    Aggregate across bars to get per-regime 'expected contribution' Sharpe.
    """
    df = bars[bars["covered"]].copy()
    edge = (df["rcvt_logret"].fillna(0) - df["bh_logret"].fillna(0)).values
    out = {}
    for k in range(3):
        w = df[f"p{k}"].fillna(0).values
        contrib = edge * w
        if contrib.std(ddof=1) < 1e-12:
            out[k] = 0.0
            continue
        out[k] = float(contrib.mean() / contrib.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="outputs/multi_asset",
                    help="Directory containing {asset}/bars.csv files")
    ap.add_argument("--output-dir", default="outputs/regime_decomp")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bar_files = sorted(in_dir.glob("*/bars.csv"))
    if not bar_files:
        print(f"ERROR: no bars.csv files in {in_dir}/")
        print("Run first: python src/multi_asset_runner.py --assets SOL,BTC,ETH --dump-bars")
        sys.exit(1)

    print(f"Found {len(bar_files)} assets with per-bar data\n")

    summary_rows = []
    soft_rows = []

    for f in bar_files:
        asset = f.parent.name
        bars = pd.read_csv(f, parse_dates=["ts"])
        pr = per_regime_stats(bars)
        conc = concentration(pr)
        sw = soft_weighted(pr, bars)

        # Headline (covered bars only)
        cov = bars[bars["covered"]].dropna(subset=["bh_logret", "rcvt_logret"])
        bh_sh = (cov["bh_logret"].mean() / cov["bh_logret"].std(ddof=1)) * np.sqrt(PERIODS_PER_YEAR)
        rcvt_sh = (cov["rcvt_logret"].mean() / cov["rcvt_logret"].std(ddof=1)) * np.sqrt(PERIODS_PER_YEAR)

        print(f"=== {asset} ===")
        print(f"  headline: BH Sharpe {bh_sh:+.2f}  RCVT Sharpe {rcvt_sh:+.2f}  Δ {rcvt_sh - bh_sh:+.2f}")
        print(f"  argmax-regime decomposition:")
        print(pr.to_string(index=False, float_format=lambda x: f"{x:+.3f}" if abs(x) > 0.001 else f"{x:.3f}"))
        print(f"  concentration (share of positive edge in strongest regime): {conc:.2f}")
        print(f"  posterior-weighted regime attribution:")
        for k, v in sw.items():
            print(f"    regime {k}: {v:+.3f} (Δ Sharpe)")
        print()

        summary_rows.append({
            "asset": asset,
            "n_test": len(bars),
            "n_covered": int(bars["covered"].sum()),
            "bh_sharpe": float(bh_sh),
            "rcvt_sharpe": float(rcvt_sh),
            "delta_sharpe": float(rcvt_sh - bh_sh),
            "concentration": conc,
            "soft_k0": sw[0],
            "soft_k1": sw[1],
            "soft_k2": sw[2],
        })

        for k, v in sw.items():
            soft_rows.append({"asset": asset, "regime": k, "soft_delta_sharpe": v})

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(soft_rows).to_csv(out_dir / "soft_attribution.csv", index=False)

    # Print the master table
    print("\n=== Master table (sorted by concentration: high = edge is regime-specific) ===")
    disp = summary[["asset", "delta_sharpe", "concentration", "soft_k0", "soft_k1", "soft_k2"]].copy()
    disp = disp.sort_values("concentration", ascending=False)
    print(disp.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))

    print(f"\nWritten: {out_dir / 'summary.csv'}")
    print(f"Written: {out_dir / 'soft_attribution.csv'}")


if __name__ == "__main__":
    main()
