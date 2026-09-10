import sys
import os
import tempfile
from pathlib import Path
# ensure the package root is on sys.path so pytest can import the local package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from volatility_backtester import VolatilityBacktester

def test_quick_outputs_generation():
    # create a small synthetic CSV dataset for the test so it does not depend on external data
    tmp = Path(tempfile.mkdtemp())
    csv_path = tmp / 'gold_test.csv'
    import pandas as pd
    dates = pd.date_range(start='2020-01-01', periods=400, freq='D')
    df = pd.DataFrame({'Date': dates, 'Close': (100 + (pd.Series(range(400)).apply(lambda x: (0.001 * x) + (0.5 * (x%10==0)))) ).values})
    df.to_csv(csv_path, index=False)

    vb = VolatilityBacktester(str(tmp), results_dir=str(Path(tempfile.gettempdir()) / "vol_test_results"))
    series = vb.load_data(pattern='gold_test').iloc[:, 0]
    feat = vb.prepare_features(series)
    train, test = vb.train_test_split(feat)
    rf, params = vb.train_random_forest(train.drop(columns=['target']), train['target'])
    vb.models_trained = {'RF': rf}
    wf = vb.walk_forward_backtest(feat, model_names=['RF'], initial_train_size=0.6, step=100)
    out_dir = vb.generate_outputs()
    p = Path(out_dir)
    assert p.exists()
    # must contain at least one PNG and the meta.json
    pngs = list(p.glob('*.png'))
    assert len(pngs) >= 1
    assert (p / 'meta.json').exists()
    # at least one MP4 animation (if ffmpeg available, otherwise skip)
    mp4s = list(p.glob('*.mp4'))
    # ensure basic metrics present
    assert 'walk_metrics' in vb.results
    assert 'RF' in vb.results['walk_metrics']
    # if an mp4 was created ensure it's non-empty
    if mp4s:
        assert mp4s[0].stat().st_size > 1000
