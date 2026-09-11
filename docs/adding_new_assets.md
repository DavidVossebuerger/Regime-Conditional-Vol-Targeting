# Adding a New Asset

To onboard a new crypto asset (or any hourly OHLC instrument) into the
risk-management pipeline:

## 1. Place the data file

Put your hourly OHLCV bars into `data/`. Two formats supported:

**CSV** (recommended for new assets):
```
data/myasset_usd_1h.csv
```
Columns: `timestamp, open, high, low, close, volume`
- `timestamp` may be either Unix milliseconds (numeric) OR ISO 8601 string
- Use any name; the loader searches by `*usd*.csv` substring

**Parquet** (for 1m data that you aggregate to 1h):
```
data/crypto_<SYMBOL>_USD_1m*.parquet
```
Must contain columns `ts` (datetime), `close` (float). Aggregated to hourly
in the loader (last close per hour).

## 2. Register the asset

Edit `src/multi_asset_runner.py`. In the `ASSETS` list near the top:

```python
ASSETS = [
    ("BTC", "parquet:1m:crypto_BTC_USD"),
    ("ETH", "parquet:1m:crypto_ETH_USD"),
    ("MYNEW", "csv:mynew_usd_1h.csv"),   # ← add this line
    ...
]
```

Format: `(SYMBOL, "csv:filename" or "parquet:1m:prefix")`.

## 3. Run

```bash
python scripts/run_pipeline.py --assets MYNEW
```

Outputs land in `outputs/MYNEW/`:
- `equity.png` — equity / rolling vol / drawdown
- `per_window.csv` — per-window OOS Sharpe
- `summary.json` — headline metrics

## 4. (Optional) Tune hyperparameters per asset

Default config (`config/default.yaml`) uses one-size-fits-all parameters
calibrated on BTC 1h. For assets with very different volatility profiles
(leveraged tokens, low-cap alts, low-volume), override per run:

```bash
python scripts/run_pipeline.py --assets MYNEW --config config/myasset.yaml
```

Where `myasset.yaml` overrides any subset of `default.yaml`:

```yaml
target_grid: [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]   # narrower range
walk_forward:
  test_size: 1440                                  # smaller windows for less data
```

## 5. (Optional) Add the asset to the default set

Edit `scripts/run_pipeline.py`:

```python
DEFAULT_ASSETS = ["BTC", "ETH", "ADA", "BNB", "DOGE", "LINK", "LTC", "SOL", "XRP", "MYNEW"]
```

## Common gotchas

| Symptom | Cause | Fix |
|---|---|---|
| "insufficient data, skipping" | Asset has <5000 hourly bars | Use a longer-history file or convert 1m → 1h |
| NaN Sharpe in output | All-zero returns (stablecoin or flat period) | Expected for stablecoins like USDC |
| All HMM windows collapse to same position | HMM convergence failed | Already retried with `implementation="log"`; otherwise try K=2 |
| Strategy underperforms B&H | Asset-specific regime pattern not captured | See `docs/results.md` § "Russell 2000" for known cases |
