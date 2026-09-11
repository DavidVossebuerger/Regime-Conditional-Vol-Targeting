"""Risk / return metrics used across the pipeline.

Single canonical implementation. Runners import from here — do NOT redefine.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def metrics(
    r: np.ndarray,
    pos: np.ndarray,
    periods_per_year: int = 8760,
    vol_window: int = 24,
    pct_within_max: float = 0.60,
) -> dict:
    """Headline metrics for a strategy-return series + position series.

    Returns dict with ann_ret, ann_vol, sharpe, sortino, max_dd, calmar, pct_within.
    All annualized. Empty / trivial series return zeros / NaN.

    Parameters
    ----------
    r : array of per-bar log returns of the strategy (incl. TC).
    pos : array of per-bar positions in [0, 1].
    periods_per_year : bars per year (8760 for hourly 24/7 crypto, 252 for daily equities).
    vol_window : rolling vol window in bars (24h for hourly crypto, 20d for daily equities).
    pct_within_max : vol-budget threshold for `pct_within` (0.60 for crypto,
        0.40 for equities — equity portfolios typically target lower vol).
    """
    if len(r) < 100 or np.std(r, ddof=1) < 1e-12:
        return {"ann_ret": 0.0, "ann_vol": 0.0, "sharpe": float("nan"),
                "sortino": float("nan"), "max_dd": 0.0, "calmar": float("nan"),
                "pct_within": float("nan")}
    sd = float(np.std(r, ddof=1))
    ann_r = float(r.mean()) * periods_per_year
    ann_v = sd * np.sqrt(periods_per_year)
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    max_dd = float(-dd.min())
    downside = r[r < 0]
    dstd = float(np.std(downside, ddof=1)) if len(downside) > 1 else float("nan")
    roll_v = pd.Series(r).rolling(vol_window).std(ddof=1).dropna() * np.sqrt(periods_per_year)
    pct_within = float((roll_v <= pct_within_max).mean()) if len(roll_v) else float("nan")
    return dict(
        ann_ret=ann_r, ann_vol=ann_v, sharpe=ann_r / ann_v,
        sortino=ann_r / dstd if dstd and dstd > 0 else float("nan"),
        max_dd=max_dd,
        calmar=ann_r / max_dd if max_dd > 0 else float("nan"),
        pct_within=pct_within,
    )


def sharpe_rank(r: np.ndarray) -> float:
    """Unscaled Sharpe for ranking thresholds during in-sample selection.

    Note: this is the *unscaled* Sharpe (no sqrt of periods), used as an
    in-sample ranking statistic on log returns where bars-per-year
    normalization would cancel across the grid search.
    """
    if len(r) < 30 or np.std(r, ddof=1) < 1e-12:
        return -np.inf
    return float(r.mean() / np.std(r, ddof=1))
