"""Quick run for VolatilityBacktester (fast settings)
- Trains a RandomForest on the gold series
- Runs a coarse walk-forward backtest (step=50)
- Generates the PDF report in results/
"""
from volatility_backtester import VolatilityBacktester
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    vb = VolatilityBacktester('..\\Kursdaten')
    data = vb.load_data(pattern='xau')
    series = data.iloc[:, 0]
    print('Loaded series length:', len(series))
    feat = vb.prepare_features(series)
    train, test = vb.train_test_split(feat)
    rf, params = vb.train_random_forest(train.drop(columns=['target']), train['target'])
    vb.models_trained = {'RF': rf}
    print('RF trained')
    wf = vb.walk_forward_backtest(feat, model_names=['RF'], initial_train_size=0.6, step=50)
    print('Walk-forward rows:', len(wf))
    out_dir = vb.generate_outputs()
    print('Outputs saved to:', out_dir)