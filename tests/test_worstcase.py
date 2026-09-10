"""Tests for the thermodynamic worst-case feature (V_wc)."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def v_wc(logret: np.ndarray, eta: float) -> float:
    """Local copy of the V_wc formula for testing."""
    z = np.log(np.mean(np.exp(eta * logret)))
    return float(z / eta)


def test_v_wc_zero_eta_limit():
    """At η → 0, V_wc → mean(r) (no perturbation)."""
    rng = np.random.default_rng(0)
    r = rng.standard_normal(500)
    v = v_wc(r, eta=1e-6)
    assert abs(v - r.mean()) < 1e-4


def test_v_wc_ge_mean():
    """V_wc ≥ E[r] always (convexity of exp)."""
    rng = np.random.default_rng(1)
    r = rng.standard_normal(1000)
    for eta in [0.5, 1.0, 2.0, 4.0]:
        v = v_wc(r, eta=eta)
        assert v >= r.mean() - 1e-9, f"V_wc violated convexity at eta={eta}: {v} < {r.mean()}"


def test_v_wc_loss_period_amplifies():
    """In a loss period (mean r < 0), V_wc with sufficient η should be > 0 and
    amplify the magnitude of mean loss."""
    rng = np.random.default_rng(2)
    # Strong loss period: -2% drift, 5% std → for η = 50, V_wc ≈ mean + η/2·var
    # ≈ -0.02 + 25 * 0.0025 = +0.0425, clearly positive.
    r = -0.02 + 0.05 * rng.standard_normal(5000)
    v = v_wc(r, eta=50.0)
    assert v > 0, f"Strong loss period at η=50 should give V_wc > 0, got {v}"
    assert v > abs(r.mean()), f"V_wc ({v}) should amplify mean loss magnitude ({r.mean()})"


def test_v_wc_converges_to_max():
    """As η → ∞, V_wc → max(r) (worst-case expected return converges to best single obs).

    Note: V_wc = log(mean(exp(ηr)))/η. At η → ∞, exp(ηr) is dominated by exp(η·max(r)),
    so mean(exp(ηr)) ≈ exp(η·max(r))/N, log(mean) ≈ η·max(r) − log(N), V_wc ≈ max(r) − log(N)/η.
    Use η large enough that log(N)/η is negligible.
    """
    r = np.array([-0.05, -0.01, 0.02, 0.03, 0.04])
    v = v_wc(r, eta=10000.0)
    # max(r) = 0.04; correction = log(5)/10000 ≈ 0.00016
    assert abs(v - r.max()) < 5e-4


def test_v_wc_larger_eta_more_conservative_in_loss():
    """Larger η should give larger V_wc in a loss period."""
    rng = np.random.default_rng(3)
    r = -0.001 + 0.01 * rng.standard_normal(2000)
    v_small = v_wc(r, eta=0.5)
    v_large = v_wc(r, eta=4.0)
    assert v_large > v_small, "Larger η should give larger worst-case loss"
