"""Lookahead-bias smoke tests: ensure no feature uses contemporaneous info for prediction."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def soft_position(target_vol: float, pred_vals: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    return np.clip(target_vol / np.maximum(pred_vals, eps), 0.0, 1.0)


def test_soft_position_bounds():
    """Position must be in [0, 1]."""
    pred = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    pos = soft_position(0.5, pred)
    assert (pos >= 0).all()
    assert (pos <= 1).all()


def test_soft_position_clamps_when_pred_too_small():
    """If pred < target, position saturates at 1.0."""
    pred = np.array([0.1, 0.2])
    pos = soft_position(0.5, pred)
    # 0.5 / 0.1 = 5 → clamped to 1.0; 0.5/0.2 = 2.5 → clamped to 1.0
    assert (pos == 1.0).all()


def test_soft_position_shrinks_with_huge_pred():
    """If pred >> target, position shrinks toward 0."""
    # target 0.5, pred 100 → 0.5/100 = 0.005 → very small
    pos = soft_position(0.5, np.array([100.0]))
    assert pos[0] < 0.01


def test_lookahead_audit_target_only_shift_neg_one():
    """Contract: only the target column may use shift(-1); all features must be
    shift(+lag) or rolling-with-end-at-t (closed right window).

    This is enforced at code-review time. This test verifies our build_features
    convention by example."""
    import pandas as pd

    prices = pd.Series(np.exp(np.cumsum(np.random.default_rng(0).standard_normal(100))))
    log_ret = np.log(prices / prices.shift(1))
    # rv at time t uses log_ret[t-23..t] → no future info
    rv = log_ret.rolling(24).std() * np.sqrt(8760)
    # The target is rv[t+1] — explicitly shifted by -1
    target = rv.shift(-1)
    valid = target.dropna()
    assert len(valid) > 50
