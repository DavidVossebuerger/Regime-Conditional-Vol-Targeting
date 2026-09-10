import pytest
import numpy as np
import matplotlib.pyplot as plt
from volatility_backtester import VolatilityBacktester


def test_fig_to_rgb():
    vb = VolatilityBacktester(data_dir='Kursdaten', results_dir='results_test')
    fig = plt.figure(figsize=(4,2))
    plt.plot([1,2,3],[1,4,9])
    img = vb._fig_to_rgb(fig)
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3 and img.shape[2] == 3
    plt.close(fig)


@pytest.mark.skipif(pytest.importorskip('prophet') is None, reason='prophet not installed')
def test_prophet_training():
    from prophet import Prophet
    import pandas as pd
    # create simple time series
    idx = pd.date_range('2020-01-01', periods=30, freq='D')
    y = np.linspace(0.1, 0.5, 30) + np.random.randn(30)*0.01
    s = pd.Series(y, index=idx)
    vb = VolatilityBacktester(data_dir='Kursdaten', results_dir='results_test')
    models = vb.train_models(s.to_frame(name='dummy').drop(columns=[]), s, method='optuna', n_trials=1, lstm_epochs=0, use_prophet=True)
    # Prophet should be in models if installed
    assert 'Prophet' in vb.models_trained or 'Prophet' in models
