# London Strategic Edge (LSE) API – Notes for Risk-Modul

Date: 2026-09-10
Author: research dump (no live code changes)

## API Overview

- **Vendor:** London Strategic Edge (https://londonstrategicedge.com)
- **What it is:** "The largest free archive of market data" – 133 B ticks, 118 k datasets, 16 k+ streamable instruments.
- **Two planes, one key:**
  - **REST (vault)** – synchronous JSON row queries for interactive pulls, async Parquet export jobs for bulk history.
  - **WebSocket** – live tick streaming with optional server-side replay (`start` parameter, up to 24 h of backfill).
- **Asset coverage:** stocks, FX, crypto, commodities, indices, ETFs, futures, options + macro/economics series (194 countries) + government bond yields.
- **Crypto coverage confirmed:** BTC/USD, ETH/USD, SOL/USD, XRP/USD return OHLCV; the marketing page also lists ADA + 50+ altcoins.
- **Crypto history depth:** tick tape back to 2017; candles extend to 1-second resolution.

## Endpoints (REST base `https://api.londonstrategicedge.com/vault`)

All REST endpoints accept `?symbol=...`, `?timeframe=...`, `?start=...`, `?end=...`, `?limit=...`, `?order=asc|desc` and require the `x-api-key` header.

| Endpoint | Method | Purpose | Notes |
|---|---|---|---|
| `/vault/usage` | GET | Plan caps + current usage (no quota cost) | **Use this as the health check.** Returns calls_per_minute, bytes caps, exports_this_hour. |
| `/vault/candles` | GET | OHLCV candles | 14 timeframes: `1s, 5s, 15s, 30s, 1m, 3m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1mo`. Capped at 5000 rows/page. |
| `/vault/ticks` | GET | Raw tick tape | Same 24 h replay as the WS plane; deep history requires an export job. |
| `/vault/datasets` | GET | Catalog of available instruments per asset class | e.g. `?category=crypto` returns the whole crypto catalog. |
| `/vault/economics` | GET | Macro time series (CPI, Fed funds, etc.) | Single series or full list. |
| `/vault/series` | GET | Government bond yields (DE10Y, US10Y, …) | |
| `/vault/options` | GET | Options chain (per underlying) | Returns ticker, IV, greeks, today's volume/premium. |
| `/vault/options_flow` | GET | Individual options prints with premium & greeks | Filter by `min_premium`, `start`, `end`. |
| `/vault/option_candles` | GET | 1-minute bars for a single contract | |
| `/vault/economic_calendar` | GET | Event calendar | |
| `/vault/insider_trades` | GET | Insider transactions | |
| `/vault/dividends`, `/vault/splits`, `/vault/cot`, `/vault/financial_reports`, `/vault/company_profiles`, `/vault/fundamentals`, `/vault/bond_yields` | GET | Reference datasets | COT uses futures codes (GC gold, CL crude, ES S&P). |
| `/vault/history` (POST/export job) | POST | Async Parquet export of deep history | Hourly budget = `exports_cap_hour`. |
| `/vault/dataset` (POST/export job) | POST | Async Parquet export of a whole reference set | Same budget. |

**WebSocket:** `wss://data-ws.londonstrategicedge.com` – auth frame `{"action":"auth","api_key":"<key>"}`, then `subscribe`/`unsubscribe` for symbols. Events: `tick`, `connected`, `authenticated`, `disconnected`, `error`.

**Legacy REST:** `https://api.londonstrategicedge.com/iso` (PostgREST grammar) still alive but not used by the current SDK.

## Auth + Rate Limits

- **Auth header:** `x-api-key: <your-key>` (lowercase, single header). The SDK also accepts it via `LSE_API_KEY` env var or `LSE(api_key=...)`.
- **Key shape:** ours is `lse_live_53ac...` (41 chars, `lse_live_` prefix) → production-tier key. Demo keys use the `lse_test_` prefix.
- **Live plan read from `/vault/usage` for our key:**

  ```json
  {"bytes_used_month":12954654,"bytes_cap_month":53687091200,
   "bytes_used_week":934615,"bytes_cap_week":16106127360,
   "exports_this_hour":0,"exports_cap_hour":5,
   "historical_data_months":-1,
   "calls_per_minute":200,"max_rows_per_request":5000,
   "vault_concurrency":2}
  ```

  Translation:
  - **200 calls/minute** (HTTP 429 after that; sliding window).
  - **50 GB / month** and **15 GB / week** allowance, shared between REST downloads **and** WS streaming. Metered by response body bytes via the `X-Data-Bytes` response header.
  - **5 Parquet export jobs / hour** (`/vault/history`, `/vault/dataset`).
  - **Unlimited historical depth** (`historical_data_months: -1`).
  - **5000 rows/page** hard cap.
  - **2 concurrent vault requests** (`vault_concurrency: 2`) – respect it, parallel calls beyond this will be rejected.

## Crypto Coverage (the key question for our use case)

**What it gives us:**
- Clean OHLCV candles for any pair in the catalog (BTC/USD, ETH/USD, SOL/USD, XRP/USD, ADA/USD, 50+ altcoins).
- Tick-level tape for replay/footprint analysis.
- Multiple timeframes down to **1 second** → fine-grained volatility / microstructure possible.
- Long history (back to 2017 ticks).

**What it does NOT give us (critical gaps for XRP failure forensics):**
- ❌ **No order book / L2 depth endpoint.** No bids/asks, no depth snapshots, no imbalance metric.
- ❌ **No sentiment scores** (no Fear&Greed, no social-volume, no funding-rate, no long/short ratio).
- ❌ **No on-chain metrics** (no wallet flows, no active addresses, no exchange inflow/outflow, no validator stats).
- ❌ **No derivatives data** (no perp funding, no OI, no liquidations) – the `/vault/options*` endpoints cover equity options, not crypto derivatives.
- ❌ **No news/event feed** tied to a symbol.

In short: **price + volume + tick microstructure only.** It is a *classical market-data feed*, not an *exchange intelligence* feed. For an XRP failure analysis it can answer *what happened on the tape* (price drop, vol spike, gap detection, drawdown stats, RV / IV-proxy via realised variance), but **not** *why it happened*.

## Sample Call (redacted)

Cheapest possible sanity call against the live API – proves the key + returns BTC/USD OHLCV:

```bash
curl -s -H "x-api-key: $LSE_KEY" \
  "https://api.londonstrategicedge.com/vault/candles?symbol=BTC/USD&timeframe=1d&order=desc&limit=2"
```

Response (live, 2026-09-10):

```json
[
  {"ts":"2026-09-10 00:00:00.000000","symbol":"BTC/USD",
   "open":78306.43,"high":78564.39,"low":76700,"close":77255,"volume":11975.27},
  {"ts":"2026-09-09 00:00:00.000000","symbol":"BTC/USD",
   "open":78455.8,"high":79760,"low":77770,"close":78306.43,"volume":14129.92}
]
```

Same shape returned for `XRP/USD`. Plan-status call (zero quota cost):

```bash
curl -s -H "x-api-key: $LSE_KEY" \
  "https://api.londonstrategicedge.com/vault/usage"
```

## Python SDK

Yes – official client on PyPI:

```bash
pip install lse-data            # core
pip install 'lse-data[frames]'  # adds pandas/pyarrow
```

```python
from lse import LSE
client = LSE()  # reads LSE_API_KEY from env
btc = client.candles("BTC/USD", "1d", start="2024-01-01")
ticks = client.history("XRP/USD", start="2025-09-01", end="2025-09-02")  # Parquet export
for tick in client.stream(["BTC/USD", "XRP/USD"]):
    print(tick.symbol, tick.price)
```

CLI also available: `lse auth lse_live_xxx` then `lse stream BTC/USD`.

## Recommendation for the risk-modul project

**Verdict: useful, but not sufficient on its own for XRP failure forensics.**

Use it for:
- **Volatility / risk metrics** on the price tape (realised vol, drawdown, VaR backtests, jump detection).
- **Cross-asset context** for the failure window (BTC/ETH/SOL behaviour around the XRP event → market-wide shock vs. idiosyncratic).
- **Intraday microstructure** down to 1 s for the exact failure timestamp.
- **Cheap, free-of-quota health check** via `/vault/usage` in any cron/monitoring loop.

Do **not** expect it to deliver:
- Order-book depth around the failure time.
- Sentiment / social / news attribution.
- On-chain attribution (wallet moves, validator behaviour, exchange flows).
- Derivatives signals (funding, OI, liquidations).

For those layers we'd still need one of: Kaiko / CryptoCompare / CoinGlass / Glassnode / Santiment (or a CEX-specific WS feed from Binance/Coinbase/Kraken for the relevant pairs). LSE can carry the price side; a second provider should carry the *why* side.

### Concrete next steps

1. Wire `LSE` into the risk-modul data layer (env var `LSE_API_KEY` already present as `LSE_KEY` in `/home/davidv/Dokumente/Risikooptimierung/.env` – add a one-line rename or `os.environ["LSE_API_KEY"] = os.environ["LSE_KEY"]` shim).
2. Build a `health.py` that calls `/vault/usage` once at startup and surfaces `bytes_used_month / bytes_cap_month` and `calls_per_minute` – this both validates the key and gives us a quota meter.
3. For the XRP failure analysis: pull `XRP/USD` 1 s and 1 m candles across the event window, plus `BTC/USD`, `ETH/USD`, `SOL/USD` for market-wide context. Compute realised vol, max drawdown, gap stats, vol-of-vol – all deterministic, no look-ahead.
4. Keep a tight loop on `X-Data-Bytes` totals (log it per call) – we share 50 GB/month with any WS streaming we add later.
