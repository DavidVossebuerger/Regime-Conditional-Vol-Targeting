"""London Strategic Edge API loader for daily equity bars.

Caches responses in `data/cache/lse_<symbol>.csv` so repeated runs don't burn quota.
"""
from __future__ import annotations
import os
import time
from pathlib import Path

import pandas as pd
import requests

DEFAULT_BASE = "https://api.londonstrategicedge.com/vault"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def _resolve_api_key() -> str | None:
    """Resolve LSE API key from env files in priority order:
    1. ~/.crypto_risk_pipeline.env
    2. .env (project)
    3. env var LSE_API_KEY
    """
    for path in [Path.home() / ".crypto_risk_pipeline.env", Path(".env")]:
        if path.exists():
            try:
                for line in path.read_text().splitlines():
                    if line.startswith("LSE_API_KEY=") and "your_" not in line:
                        return line.split("=", 1)[1].strip()
                    if line.startswith("LSE_KEY=") and "your_" not in line:
                        return line.split("=", 1)[1].strip()
            except OSError:
                pass
    return os.environ.get("LSE_API_KEY") or os.environ.get("LSE_KEY")


def load_lse_daily(symbol: str, base: str = DEFAULT_BASE, use_cache: bool = True) -> pd.Series | None:
    """Fetch daily OHLC bars from LSE for the given symbol. Returns a pd.Series of close prices."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"lse_{symbol.replace('/', '_')}.csv"

    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file)
        df["Date"] = pd.to_datetime(df["ts"])
        return df.set_index("Date").sort_index()["close"].astype(float)

    key = _resolve_api_key()
    if not key:
        print("  LSE_API_KEY not found in ~/.crypto_risk_pipeline.env, .env, or env vars")
        return None

    hdr = {"x-api-key": key}
    url = f"{base}/candles"
    try:
        r = requests.get(url, params={"symbol": symbol, "timeframe": "1d", "limit": 5000},
                         headers=hdr, timeout=30)
    except requests.RequestException as e:
        print(f"  LSE request failed for {symbol}: {e}")
        return None

    if r.status_code != 200:
        print(f"  LSE returned {r.status_code} for {symbol}: {r.text[:100]}")
        return None

    try:
        data = r.json()
    except ValueError:
        print(f"  LSE response not JSON for {symbol}")
        return None

    if not isinstance(data, list) or not data:
        print(f"  LSE empty response for {symbol}")
        return None

    df = pd.DataFrame(data)
    if "ts" not in df.columns or "close" not in df.columns:
        print(f"  LSE unexpected schema for {symbol}: {list(df.columns)}")
        return None

    df["Date"] = pd.to_datetime(df["ts"])
    df = df.set_index("Date").sort_index()
    if use_cache:
        df.reset_index().to_csv(cache_file, index=False)
    return df["close"].astype(float)


def clear_cache() -> None:
    """Remove all cached LSE responses."""
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("lse_*.csv"):
            f.unlink()
