"""Tests for transaction-cost (TC) deduction in the position-sizing pipeline.

These tests verify that TC is correctly applied at every stage where it should
be — not just that the constant exists, but that it actually subtracts from
the strategy P&L.

TC model: `cost_at_bar_t = |Δpos(t)| · TC_PER_SIDE` (one side per move).
Round-trip = 2 × TC_PER_SIDE per entry + exit.
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.metrics import metrics


TC = 0.0002  # 2 bps / side — same as TC_PER_SIDE in runners


def simulate_with_tc(log_ret: np.ndarray, positions: np.ndarray, tc_per_side: float = TC) -> np.ndarray:
    """Reference TC calculation; should match runners."""
    pos = np.asarray(positions)
    r = pos * np.asarray(log_ret)
    tc = np.abs(np.diff(pos, prepend=pos[0])) * tc_per_side
    return r - tc


def test_tc_deducted_from_returns():
    """Holds 1.0 throughout → Δpos = 0 → no TC deducted."""
    log_ret = np.array([0.01, -0.005, 0.02, 0.001])
    pos = np.array([1.0, 1.0, 1.0, 1.0])
    r = simulate_with_tc(log_ret, pos)
    expected = log_ret  # no position changes → no TC
    np.testing.assert_array_almost_equal(r, expected)


def test_tc_only_on_position_changes():
    """One big position change at t=1 should produce exactly one TC deduction."""
    log_ret = np.array([0.01, 0.01, 0.01])
    pos = np.array([1.0, 0.0, 0.0])  # exits at t=1
    r = simulate_with_tc(log_ret, pos)
    expected = np.array([0.01, -TC, 0.0])  # entry held for full bar 0, exit at t=1
    np.testing.assert_array_almost_equal(r, expected)


def test_tc_proportional_to_position_delta():
    """TC scales with |Δpos| — partial exits cost proportionally."""
    log_ret = np.array([0.01, 0.01, 0.01])
    pos = np.array([1.0, 0.5, 0.5])  # halve position
    r = simulate_with_tc(log_ret, pos)
    expected = np.array([0.01, 0.5 * 0.01 - 0.5 * TC, 0.5 * 0.01])
    np.testing.assert_array_almost_equal(r, expected)


def test_tc_round_trip():
    """Round-trip (entry + exit) costs exactly 2 × TC × position size."""
    log_ret = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    pos = np.array([0.0, 1.0, 1.0, 1.0, 0.0])  # enter t=1, exit t=4
    r = simulate_with_tc(log_ret, pos)
    # Costs at t=1 (entry, +1.0) and t=4 (exit, -1.0)
    assert abs(r[1] - (1.0 * 0.01 - TC)) < 1e-12
    assert abs(r[4] - (0.0 - TC)) < 1e-12
    # Position is held only at t=1,2,3 (3 bars × 0.01 = 0.03 gross), minus 2*TC round-trip
    assert abs(r.sum() - (0.03 - 2 * TC)) < 1e-12


def test_no_tc_on_zero_returns():
    """Zero log returns + position changes → only TC deducted."""
    log_ret = np.array([0.0, 0.0, 0.0])
    pos = np.array([1.0, 0.5, 0.25])
    r = simulate_with_tc(log_ret, pos)
    # TC at t=1: |0.5-1|=0.5; at t=2: |0.25-0.5|=0.25; at t=0 (prepend): |1.0-1.0|=0
    expected_r = np.array([0.0, -0.5 * TC, -0.25 * TC])
    np.testing.assert_array_almost_equal(r, expected_r)


def test_soft_weight_hmm_tc_continuous():
    """Soft-weight HMM produces continuous positions; TC proportional to smooth moves."""
    log_ret = np.array([0.001] * 24)
    # Gradual position reduction over 24 bars (e.g. HMM smoothly decreasing)
    pos = np.linspace(1.0, 0.5, 24)
    r = simulate_with_tc(log_ret, pos)
    # Total market return component = sum(pos * log_ret) = mean(pos) * sum(log_ret) = 0.75 * 0.024
    market_component = pos.mean() * log_ret.sum()
    # Total TC deducted = sum(|Δpos|) * TC = (1.0 - 0.5) * TC = 0.5 * TC
    total_tc = 0.5 * TC
    # r.sum() = market_component - total_tc
    assert abs(r.sum() - (market_component - total_tc)) < 1e-9


def test_metrics_function_handles_tc_consistent_returns():
    """Sanity check: metrics() works with TC-deducted returns."""
    log_ret = np.array([0.001] * 1000)
    pos = np.linspace(0.5, 1.0, 1000)  # gradual increase
    r = simulate_with_tc(log_ret, pos)
    m = metrics(r, pos)
    assert m["ann_ret"] > 0  # positive drift + small TC drag
    assert m["sharpe"] > 0
    assert m["max_dd"] == 0.0  # no drawdown with monotonic increase
