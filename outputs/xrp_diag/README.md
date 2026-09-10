This folder contains a template implementation of a Volatility Backtester designed to:

- Load prices from CSVs in `Kursdaten/`
- Prepare features and targets (realized volatility)
- Train RandomForest, XGBoost, and an LSTM (Keras) model
- Run walk-forward backtesting and compute evaluation metrics
- Generate visual outputs (PNGs, GIFs, MP4s) in `results/outputs_YYYYMMDD_HHMMSS` instead of a single PDF

Quick start:

1. Install dependencies (recommended in a virtualenv):

   pip install -r requirements_dev.txt

2. Run the example (quick RF-run):

   python quick_run.py

Outputs

- `generate_outputs()` creates a timestamped folder with the following artefacts:
  - `01_executive_summary.png` and `01_executive_summary.txt`
  - `02_model_comparison.png` (table+RMSE chart)
  - `03_realized_vs_predicted.png`
  - `04_forecast_errors.png`, `04_rolling_rmse.png`, `04_rolling_rmse.gif`, `04_rolling_rmse.mp4`
  - `05_error_histogram.png`, `05_qq_plot.png`, `05_pred_vs_real_scatter.png`
  - `06_rf_feature_importance.png` (if RF present)
  - `07_monthly_MAE_<model>.png` + `07_monthly_MAE_<model>.gif/mp4`
  - `08_rmse_evolution.gif/mp4` (animated RMSE by model over months)
  - `09_pred_vs_real_evolution.gif/mp4` (animated scatter over time)
  - `meta.json` with summary metadata (best model, metrics)

Notes & animations

- GIFs and MP4s are created when possible. MP4 generation uses `imageio` with ffmpeg or `moviepy` as fallback; if neither is available GIFs are created (or fallback to images).
- The visual theme uses `seaborn-darkgrid` with consistent fonts and sizes.

Using the existing Volatility model from `Option_Pricer`:

You can attach and use the copied `VolatilityModel` (from `Option_Pricer/hydra-pricer/src/models/volatility.py`) with the backtester:

```python
from volatility_backtester import VolatilityBacktester
vb = VolatilityBacktester('Kursdaten')
vb.attach_volatility_model()
series = vb.load_data(pattern='xau').iloc[:,0]
vb.fit_external_models(series, fit_hmm=True, fit_nn=False)
forecast = vb.forecast_external(steps=5)
print(forecast)
```

This uses the copied `volatility_model.py` in this folder and provides GARCH/HAR/HMM/NN functionality for forecasting and ensemble composition.