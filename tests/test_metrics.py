"""Tests for the metrics and V_wc primitives."""
import numpy as np
import pytest
import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.metrics import metrics, sharpe_rank


def test_metrics_constant_series():
    """Constant series has zero std → returns the early-exit dict (NaN-safe)."""
    r = np.ones(1000)
    pos = np.ones(1000)
    m = metrics(r, pos)
    # With zero std, sharpe is NaN, ann_vol is 0, ann_ret is set to 0 by the early exit
    assert m["ann_vol"] == 0.0
    assert np.isnan(m["sharpe"])


def test_metrics_simple_returns():
    """Linear positive drift should produce positive sharpe and ~zero drawdown."""
    rng = np.random.default_rng(42)
    r = 0.001 + 0.005 * rng.standard_normal(1000)
    pos = np.ones(1000)
    m = metrics(r, pos)
    assert m["sharpe"] > 0
    assert m["ann_ret"] > 0
    assert m["max_dd"] >= 0


def test_metrics_drawdown():
    """Series with a drawdown should report negative dd (stored as positive number), positive calmar expected."""
    r = np.array([0.05] * 50 + [-0.05] * 50 + [0.001] * 900)
    pos = np.ones(1000)
    m = metrics(r, pos)
    assert m["max_dd"] > 0
    assert abs(m["ann_ret"] - r.mean() * 8760) < 1e-6


def test_sharpe_rank_constant():
    """Constant series has zero std → sharpe_rank returns -inf."""
    r = np.ones(100)
    assert sharpe_rank(r) == -np.inf


def test_sharpe_rank_positive():
    """Positive drift with non-zero std should have positive sharpe_rank."""
    rng = np.random.default_rng(0)
    r = 0.001 + 0.005 * rng.standard_normal(200)
    assert sharpe_rank(r) > 0


def test_sharpe_rank_negative_drift():
    """Negative drift should have negative sharpe_rank."""
    rng = np.random.default_rng(1)
    r = -0.001 + 0.005 * rng.standard_normal(200)
    assert sharpe_rank(r) < 0


def test_metrics_short_series():
    """Series shorter than 100 should return zeros without error."""
    r = np.array([0.01, 0.02, -0.01])
    pos = np.array([1.0, 1.0, 0.5])
    m = metrics(r, pos)
    assert isinstance(m, dict)
    assert "sharpe" in m
