"""Example driver script for the VolatilityBacktester"""
from volatility_backtester import VolatilityBacktester

if __name__ == '__main__':
    vb = VolatilityBacktester('..\\Kursdaten', 'results')
    data = vb.load_data(pattern='xau')
    print('Loaded data shape:', data.shape)
    series = data.iloc[:,0]
    feat = vb.prepare_features(series, window=20, lags=5)
    train, test = vb.train_test_split(feat, split=0.8)
    X_train = train.drop(columns=['target'])
    y_train = train['target']
    X_test = test.drop(columns=['target'])
    y_test = test['target']

    rf, params = vb.train_random_forest(X_train, y_train)
    xgb = vb.train_xgboost(X_train, y_train)
    preds = vb.predict_model(rf, X_test)
    print('RF eval:', vb._eval_metrics(y_test.values, preds))
    res = vb.walk_forward(feat, feat)
    print('Walk-forward metrics:', vb.evaluate_walk())
    report_file = vb.generate_report()
    print('Report generated at', report_file)
