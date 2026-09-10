"""Risk / return metrics used across the pipeline."""
from __future__ import annotations
import numpy as np
import pandas as pd


def metrics(r: np.ndarray, pos: np.ndarray, periods_per_year: int = 8760, vol_window: int = 24) -> dict:
    """Headline metrics for a strategy-return series + position series.

    Returns dict with ann_ret, ann_vol, sharpe, sortino, max_dd, calmar, pct_within.
    All annualized. Empty / trivial series return zeros / NaN.
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
    pct_within = float((roll_v <= 0.60).mean()) if len(roll_v) else float("nan")
    return dict(
        ann_ret=ann_r, ann_vol=ann_v, sharpe=ann_r / ann_v,
        sortino=ann_r / dstd if dstd and dstd > 0 else float("nan"),
        max_dd=max_dd,
        calmar=ann_r / max_dd if max_dd > 0 else float("nan"),
        pct_within=pct_within,
    )


def sharpe_rank(r: np.ndarray) -> float:
    """Unscaled Sharpe for ranking thresholds during in-sample selection."""
    if len(r) < 30 or np.std(r, ddof=1) < 1e-12:
        return -np.inf
    return float(r.mean() / np.std(r, ddof=1))
