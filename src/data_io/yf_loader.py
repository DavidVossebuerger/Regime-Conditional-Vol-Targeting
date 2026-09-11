"""yfinance data loader with local CSV caching.

yfinance is free / no licensing restrictions → caching is fine (unlike LSE).
Returns pd.Series of close prices indexed by date.

Usage:
    from data_io.yf_loader import fetch
    s = fetch("BTC-USD", interval="1d", period="max")
    s = fetch("AAPL", interval="1d", period="max")
"""
from __future__ import annotations
from pathlib import Path
import time
import pandas as pd

import yfinance as yf

CACHE_DIR = Path("data/yfinance")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_name(ticker: str) -> str:
    return ticker.replace("/", "-").replace("=", "_").replace(".", "_")


def fetch(ticker: str, period: str = "max", interval: str = "1d",
         cache: bool = True, retries: int = 2) -> pd.Series | None:
    """Fetch close prices for `ticker` via yfinance.

    Parameters
    ----------
    ticker : yfinance ticker (e.g. "BTC-USD", "AAPL", "RIOT")
    period : "max", "10y", "5y", etc.
    interval : "1d", "1h", "1m" (note: 1m/1h limited to last ~60 days for 1m, ~2y for 1h)
    cache : if True (default), read/write local CSV cache at data/yfinance/

    Returns
    -------
    pd.Series of close prices indexed by date, or None on failure.
    """
    cache_path = CACHE_DIR / f"{_safe_name(ticker)}_{interval}.csv"
    if cache and cache_path.exists():
        try:
            df = pd.read_csv(cache_path, parse_dates=["Date"]).set_index("Date")
            s = df["close"].astype(float).sort_index()
            if len(s) >= 50:
                return s
        except Exception:
            pass  # cache corrupt, refetch

    last_err = None
    for attempt in range(retries + 1):
        try:
            data = yf.download(ticker, period=period, interval=interval,
                               progress=False, auto_adjust=True,
                               threads=False)
            break
        except Exception as e:
            last_err = e
            time.sleep(1.0 + attempt)
    else:
        print(f"  yf download failed for {ticker}: {last_err}")
        return None

    if data is None or data.empty or len(data) < 50:
        print(f"  yf empty/short for {ticker}: {0 if data is None else len(data)} rows")
        return None

    # yfinance ≥0.2 returns multi-level columns when batched; single ticker may also
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Close" not in data.columns:
        print(f"  no Close for {ticker}: {list(data.columns)}")
        return None

    s = data["Close"].astype(float)
    s.index.name = "Date"
    s = s.dropna()

    if cache:
        try:
            s.to_frame("close").to_csv(cache_path)
        except Exception as e:
            print(f"  cache write failed for {ticker}: {e}")

    return s


def bulk_fetch(tickers: list[str], period: str = "max", interval: str = "1d",
               sleep: float = 0.1, log_every: int = 10) -> dict[str, pd.Series]:
    """Fetch a batch of tickers with progress bar + ETA. Returns {ticker: pd.Series}."""
    out = {}
    failures = []
    n = len(tickers)
    if n == 0:
        return out
    t_start = time.time()
    last_print = 0.0
    for i, t in enumerate(tickers, 1):
        s = fetch(t, period=period, interval=interval)
        if s is not None and len(s) >= 100:
            out[t] = s
        else:
            failures.append(t)
        # Periodic progress line — every 10 tickers or every 3 seconds
        now = time.time()
        if i % log_every == 0 or i == n or (now - last_print) > 3.0:
            elapsed = now - t_start
            avg = elapsed / i
            eta = avg * (n - i)
            rate = i / max(elapsed, 0.01)
            pct = 100 * i / n
            bar_w = 30
            filled = int(bar_w * i / n)
            bar = "█" * filled + "░" * (bar_w - filled)
            print(
                f"  [{bar}] {pct:5.1f}% {i}/{n}  "
                f"ok={len(out):3d} fail={len(failures):3d}  "
                f"{rate:.1f} t/s  "
                f"elapsed={int(elapsed)}s  ETA={int(eta)}s",
                flush=True,
            )
            last_print = now
        time.sleep(sleep)
    if failures:
        print(f"  failed tickers: {failures[:30]}{' ...' if len(failures) > 30 else ''}")
    return out
