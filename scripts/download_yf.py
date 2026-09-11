"""Batch-download yfinance data for the full test universes.

Caches per-ticker CSVs in data/yfinance/. Idempotent — re-running skips already
cached files. yfinance has no licensing restrictions, so local caching is OK.

Usage:
    python scripts/download_yf.py --universe all
    python scripts/download_yf.py --universe crypto
    python scripts/download_yf.py --universe equities
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_io.yf_loader import bulk_fetch  # noqa: E402

# 24 crypto with deep history, mix of high-vol degen + majors
CRYPTO_TICKERS = [
    # Original 9 (with refreshed daily history)
    "BTC-USD", "ETH-USD", "ADA-USD", "BNB-USD", "DOGE-USD",
    "LINK-USD", "LTC-USD", "SOL-USD", "XRP-USD",
    # New high-vol additions
    "AVAX-USD", "DOT-USD", "MATIC-USD", "ATOM-USD", "NEAR-USD",
    "APT-USD", "SUI-USD", "FTM-USD", "ETC-USD", "XMR-USD",
    "DASH-USD", "BCH-USD", "EOS-USD", "ALGO-USD",
]


def _yfinance_ticker(raw: str) -> str:
    """Convert Wikipedia-style ticker to yfinance format (BRK.B → BRK-B)."""
    return str(raw).strip().upper().replace(".", "-").replace(" ", "-")


def _scrape_wikipedia(url: str, table_index: int = 0) -> list[str]:
    """Scrape a list of tickers from a Wikipedia index constituents page."""
    print(f"  fetching {url}")
    try:
        # Wikipedia requires a real User-Agent (otherwise 403)
        tables = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})
    except Exception as e:
        print(f"    FAILED: {e}")
        return []
    if table_index >= len(tables):
        print(f"    table[{table_index}] not found, only {len(tables)} tables")
        return []
    df = tables[table_index]
    # Try common ticker column names
    for col in ["Symbol", "Ticker", "symbol", "ticker"]:
        if col in df.columns:
            return [_yfinance_ticker(t) for t in df[col].dropna().tolist()]
    print(f"    no ticker column found in {list(df.columns)[:6]}")
    return []


def _fetch_csv(url: str) -> pd.DataFrame | None:
    """Fetch a CSV from a URL with proper UA."""
    import io, requests
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        if r.status_code != 200:
            print(f"    HTTP {r.status_code}: {url}")
            return None
        return pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        print(f"    FAILED: {e}")
        return None


def load_sp500_tickers() -> list[str]:
    """S&P 500 — use the well-maintained `datasets/s-and-p-500-companies` GitHub mirror."""
    print("  fetching S&P 500 (GitHub datasets mirror)")
    df = _fetch_csv("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv")
    if df is None or "Symbol" not in df.columns:
        return _scrape_wikipedia("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", 0)
    return [_yfinance_ticker(t) for t in df["Symbol"].dropna().tolist()]


def load_sp400_tickers() -> list[str]:
    """S&P MidCap 400 — scrape from Wikipedia with User-Agent."""
    return _scrape_wikipedia("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", 0)


def load_russell_tickers() -> list[str]:
    """Load Russell 2000 tickers from the local CSV (column 'ticker')."""
    r2k = Path("data/russell2000_top200.csv")
    if not r2k.exists():
        print(f"WARNING: {r2k} not found, skipping Russell 2000")
        return []
    df = pd.read_csv(r2k)
    return [_yfinance_ticker(t) for t in df["ticker"].dropna().tolist()]


def load_equity_universe() -> list[str]:
    """S&P 500 + S&P MidCap 400 + Russell 2000 ≈ ~1100 unique tickers."""
    print("Loading equity index constituents from Wikipedia + local CSV...")
    sp500 = load_sp500_tickers()
    sp400 = load_sp400_tickers()
    r2k = load_russell_tickers()
    print(f"  S&P 500:      {len(sp500)} tickers")
    print(f"  S&P MidCap:   {len(sp400)} tickers")
    print(f"  Russell 2000: {len(r2k)} tickers")
    seen = set()
    out = []
    for src in (sp500, sp400, r2k):
        for t in src:
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=["crypto", "equities", "all"], default="all")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--period", default="max")
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--limit-equities", type=int, default=None,
                    help="Optional cap on equity tickers (for testing)")
    args = ap.parse_args()

    tickers: list[str] = []
    if args.universe in ("crypto", "all"):
        tickers.extend(CRYPTO_TICKERS)
    if args.universe in ("equities", "all"):
        eq = load_equity_universe()
        if args.limit_equities:
            eq = eq[:args.limit_equities]
            print(f"  capped to {len(eq)} equities")
        tickers.extend(eq)

    # Deduplicate but preserve order
    seen = set()
    tickers = [t for t in tickers if not (t in seen or seen.add(t))]

    print(f"\n=== yfinance batch download ===")
    print(f"  universe:  {args.universe}")
    print(f"  interval:  {args.interval}")
    print(f"  period:    {args.period}")
    print(f"  tickers:   {len(tickers)}")
    print(f"  cache dir: data/yfinance/\n")

    t0 = time.time()
    result = bulk_fetch(tickers, period=args.period, interval=args.interval, sleep=args.sleep)
    dt = time.time() - t0

    print(f"\n=== Done in {int(dt)}s ===")
    print(f"  successful: {len(result)}/{len(tickers)}")
    if len(result) > 0:
        lens = sorted(len(s) for s in result.values())
        print(f"  history:    min {lens[0]}d, median {lens[len(lens)//2]}d, max {lens[-1]}d")


if __name__ == "__main__":
    main()

