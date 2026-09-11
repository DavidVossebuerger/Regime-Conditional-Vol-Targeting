"""CLI wrapper for the crypto RCVT pipeline.

Spawns `src/multi_asset_runner.py` once per asset to get clean per-asset
progress + ETA. Pass `--tc-per-side` to forward to each invocation; bootstrap
iterations are controlled inside the runner via its own constants
(`N_BOOT`, `BLOCK_SIZE`).

Usage:
    python scripts/run_pipeline.py --assets BTC,ETH,XRP
    python scripts/run_pipeline.py --tc-per-side 0.0005
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ASSETS = ["BTC", "ETH", "ADA", "BNB", "DOGE", "LINK", "LTC", "SOL", "XRP"]


def parse_args():
    ap = argparse.ArgumentParser(description="RCVT crypto pipeline (per-asset wrapper)")
    ap.add_argument("--assets", default=",".join(DEFAULT_ASSETS),
                    help="Comma-separated asset list")
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs/multi_asset"))
    ap.add_argument("--tc-per-side", type=float, default=None,
                    help="TC per side as decimal (e.g. 0.0002 = 2 bps). "
                         "Forwarded to each per-asset runner invocation.")
    return ap.parse_args()


def run_one_asset(asset: str, args, idx: int, total: int) -> tuple[bool, float]:
    """Run multi_asset_runner.py for a single asset; return (success, elapsed_seconds)."""
    print(f"\n[{idx}/{total}] {asset}")
    asset_dir = Path(args.output_dir) / asset
    asset_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "multi_asset_runner.py"),
        "--assets", asset,
        "--output-dir", args.output_dir,
    ]
    if args.tc_per_side is not None:
        cmd += ["--tc-per-side", str(args.tc_per_side)]

    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    dt = time.time() - t0
    return (rc == 0), dt


def main():
    args = parse_args()
    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    n = len(assets)
    print(f"\n=== RCVT Crypto Pipeline ===")
    print(f"  assets:    {n} ({', '.join(assets)})")
    print(f"  output:    {args.output_dir}")
    if args.tc_per_side is not None:
        print(f"  tc/side:   {args.tc_per_side*1e4:.2f} bps")
    print(f"  started:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    start = time.time()
    successes = []
    failures = []
    for i, asset in enumerate(assets, 1):
        ok, dt = run_one_asset(asset, args, i, n)
        if ok:
            successes.append((asset, dt))
        else:
            failures.append((asset, dt))
        elapsed = time.time() - start
        avg = elapsed / i
        eta = avg * (n - i)
        eta_str = f"{int(eta // 60)}m{int(eta % 60):02d}s"
        print(f"  [{i}/{n}] {asset} done in {int(dt)}s   "
              f"(elapsed {int(elapsed)}s, ETA {eta_str})")

    total = time.time() - start
    print(f"\n=== Done in {int(total // 60)}m{int(total % 60):02d}s ===")
    print(f"  {len(successes)}/{n} assets succeeded")
    if failures:
        print(f"  failed: {[a for a, _ in failures]}")
    print(f"\n  outputs:")
    for asset, _ in successes:
        out = Path(args.output_dir) / asset
        png = out / "equity.png"
        summ = out / "summary.json"
        if png.exists():
            print(f"    {asset}: {png.relative_to(PROJECT_ROOT)} ({png.stat().st_size} bytes)")
        if summ.exists():
            print(f"    {asset}: {summ.relative_to(PROJECT_ROOT)} ({summ.stat().st_size} bytes)")

    cross = Path(args.output_dir) / "cross_asset.png"
    summary_csv = Path(args.output_dir) / "per_asset_summary.csv"
    if cross.exists():
        print(f"\n  cross-asset plot: {cross.relative_to(PROJECT_ROOT)}")
    if summary_csv.exists():
        print(f"  summary CSV:      {summary_csv.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
