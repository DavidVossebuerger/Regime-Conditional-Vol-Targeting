"""Run the equities pipeline on just the major indices (no need to re-run 1000+ equities).

Tests whether RCVT edge holds on S&P 500, NASDAQ, Dow Jones, etc.
Uses anchored WF (strictest no-lookahead test).
"""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from equities_runner import run_one_symbol

# Major US + international indices (all in data/yfinance/)
INDICES = [
    "^GSPC", "^SPX", "^DJI", "^IXIC", "^NDX",
    "^RUT", "^OEX", "^MID", "^NYA", "^XAX",
    "^STOXX", "^GDAXI", "^N225", "^FTSE", "^HSI",
]


def main():
    out_dir = Path("outputs/indices_anchored")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Running RCVT on {len(INDICES)} indices (anchored WF) ===\n")
    rows = []
    failed = []
    for sym in INDICES:
        try:
            r = run_one_symbol(sym, out_dir, data_source="yf",
                               dump_bars=False, anchored_wf=True)
            if r is not None:
                rows.append(r)
                print(f"  ✓ {sym:<8s} ΔSharpe={r['headline_hmm']['sharpe']-r['headline_bh']['sharpe']:+.3f} "
                      f"p_beats={r['p_hmm_beats_bh']:.2f}")
            else:
                failed.append(sym)
                print(f"  ✗ {sym:<8s} skipped")
        except Exception as e:
            failed.append(sym)
            print(f"  ✗ {sym:<8s} ERROR: {e}")

    # Write summary
    if rows:
        import pandas as pd
        df = pd.DataFrame([{
            "symbol": r["symbol"], "n_test_bars": r["n_test_bars"],
            "n_windows": r["n_windows"],
            "bh_sharpe": r["headline_bh"]["sharpe"],
            "hmm_sharpe": r["headline_hmm"]["sharpe"],
            "sharpe_delta": r["headline_hmm"]["sharpe"] - r["headline_bh"]["sharpe"],
            "bh_max_dd": r["headline_bh"]["max_dd"],
            "hmm_max_dd": r["headline_hmm"]["max_dd"],
            "p_hmm_beats_bh": r["p_hmm_beats_bh"],
        } for r in rows]).sort_values("sharpe_delta", ascending=False)
        df.to_csv(out_dir / "indices_summary.csv", index=False)
        print(f"\n=== Summary ({len(df)} indices) ===")
        print(df.round(3).to_string(index=False))
        print(f"\nMean ΔSharpe: {df['sharpe_delta'].mean():+.4f}")
        print(f"Hit rate: {(df['sharpe_delta']>0).mean()*100:.1f}%")
    if failed:
        print(f"\nFailed: {failed}")


if __name__ == "__main__":
    main()
