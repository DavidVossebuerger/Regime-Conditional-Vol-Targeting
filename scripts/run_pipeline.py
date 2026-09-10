"""CLI entrypoint for the crypto risk-management pipeline.

Spawns multi_asset_runner.py per asset to get clean per-asset progress + ETA.

Usage:
    python scripts/run_pipeline.py --assets BTC,ETH,XRP
    python scripts/run_pipeline.py --config config/custom.yaml --n-boot 5000
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
    ap = argparse.ArgumentParser(description="Crypto risk-management pipeline")
    ap.add_argument("--assets", default=",".join(DEFAULT_ASSETS),
                    help="Comma-separated asset list")
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-wc", action="store_true",
                    help="Disable worst-case feature")
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
    env_extra = {
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
    }
    import os
    env = {**os.environ, **env_extra}
    t0 = time.time()
    rc = subprocess.run(cmd, env=env).returncode
    dt = time.time() - t0
    return (rc == 0), dt


def main():
    args = parse_args()
    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    n = len(assets)
    print(f"\n=== Crypto Risk-Management Pipeline ===")
    print(f"  assets:    {n} ({', '.join(assets)})")
    print(f"  output:    {args.output_dir}")
    print(f"  wc-feat:   {not args.no_wc}")
    print(f"  n-boot:    {args.n_boot}")
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
        # ETA based on average so far
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

    # Final cross-asset summary location
    cross = Path(args.output_dir) / "cross_asset.png"
    summary_csv = Path(args.output_dir) / "per_asset_summary.csv"
    if cross.exists():
        print(f"\n  cross-asset plot: {cross.relative_to(PROJECT_ROOT)}")
    if summary_csv.exists():
        print(f"  summary CSV:      {summary_csv.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
