# Data directory

This directory is git-ignored. Place your input OHLC bars here before running the
pipeline.

## Crypto (hourly bars)

The crypto pipeline (`src/multi_asset_runner.py`) looks for:

- **Parquet** (`crypto_BTC_USD_1m*.parquet`, `crypto_ETH_USD_1m*.parquet`): 1-minute bars
  with columns `ts` (datetime), `close` (float). Aggregated to 1h internally.
- **CSV** (`<symbol>_usd_1h.csv`): hourly bars with columns
  `timestamp, open, high, low, close, volume`. Timestamp may be Unix milliseconds
  or ISO 8601 string.

You need at minimum: BTC and ETH parquet files (for the deep run) and 8 alt-coin
CSVs (for the multi-asset comparison). Sample sources:

- [Binance Vision](https://data.binance.vision/) — historical klines export
- [CryptoDataDownload](https://www.cryptodatadownload.com/) — aggregated CSVs
- [LSE API](https://londonstrategicedge.com/api-documentation/) — see `docs/lse_api_notes.md`

## Equities (daily bars)

The equities pipeline (`src/equities_runner.py`) fetches daily bars from the
London Strategic Edge API on demand and caches them in `data/cache/`. Set your
`LSE_API_KEY` in `.env` (copy `.env.example`).

## XAU legacy data

Gold/USD daily bars for the legacy XAU pipeline (now in `.bak/legacy_xau_daily/`).

## Cache

`data/cache/lse_*.csv` — cached LSE API responses. Safe to delete (re-fetched on
next run).
