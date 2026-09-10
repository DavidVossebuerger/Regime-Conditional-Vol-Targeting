import pytest
from volatility_backtester import CLIProgress, compute_est_total
import sys


def test_cli_progress_caps(capsys):
    p = CLIProgress()
    p.start(10)
    # advance beyond total
    p.advance(15, 'Over')
    # capture the printed status
    p._print_status('Check')
    captured = capsys.readouterr()
    assert '100.00%' in captured.out or '100.00%' in captured.err
    assert 'ETA: 00:00:00' in captured.out


def test_compute_estimator_basic():
    # no optuna available path still returns deterministic number
    total = compute_est_total(n_samples=1000, n_trials=50, use_xgb=True, use_lstm=True, lstm_epochs=20, use_prophet=True, step_size=50)
    # Compute expected value depending on whether optuna is available
    import importlib
    optuna_available = importlib.util.find_spec('optuna') is not None
    rf_trials = 50 if optuna_available else 1
    xgb_trials = rf_trials if True else 0
    expected = rf_trials + xgb_trials + 1 + 20 + 1 + 2 + 1 + 8 + 1
    assert total == expected
