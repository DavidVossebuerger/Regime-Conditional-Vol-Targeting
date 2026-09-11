# Crypto Risk-Management Pipeline

[![Tests](https://github.com/DavidVossebuerger/Risk-Management/actions/workflows/tests.yml/badge.svg)](https://github.com/DavidVossebuerger/Risk-Management/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A regime-aware risk-management system for crypto **and** equities. Combines a
**Random Forest vol forecast** with a **Gaussian HMM regime classifier** to
produce a soft-vol-targeted position sizing rule that outperforms Buy-and-Hold
on a broad set of crypto and US-equity assets with strict walk-forward validation
and block-bootstrap significance.

> **Architecture (canonical config):**
> - **Random Forest** predicts next-period realized vol from 17 lagged features
> - **Gaussian HMM** (K=3) classifies each bar into regime probabilities
> - **Per-state vol-target** chosen per walk-forward window via joint brute-force
> - **Position**: `pos(t) = Σ_k P(state=k|t) · clip(target_k / pred_vol(t+1), 0, 1)`
> - **TC**: 2 bps/side · **Block bootstrap**: 60-day blocks · 1000–2000 iters

## TL;DR

```bash
git clone https://github.com/DavidVossebuerger/Risk-Management.git
cd Risk-Management
make install
make run-crypto       # full crypto pipeline (~30 min)
make run-equities     # US small-caps via LSE (need LSE_API_KEY)
make run-russell      # Russell 2000 top-200 (~10 min with cache)
make test             # 11 unit tests
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify environment
python -c "import numpy, pandas, sklearn, hmmlearn, matplotlib; print('ok')"

# 3. (Optional) Live data via London Strategic Edge API
cp .env.example .env
# edit .env and set LSE_API_KEY=...

# 4a. Crypto pipeline (hourly bars from local data/*.csv|parquet)
python scripts/run_pipeline.py --assets BTC,ETH,ADA,BNB,DOGE,LINK,LTC,SOL,XRP

# 4b. Equities pipeline (daily bars from LSE API)
python src/equities_runner.py

# 5. Run unit tests
pytest tests/ -v
```

## Glossary

| Term | Meaning |
|---|---|
| **Sharpe** | annualized mean return / annualized vol of returns. Risk-adjusted return. |
| **Sortino** | like Sharpe but only downside std in denominator. |
| **Max DD** | maximum peak-to-trough drawdown (positive number, log-return). Smaller is better. |
| **Calmar** | annualized return / Max DD. |
| **B&H** | Buy-and-Hold baseline — always fully invested, no transaction costs. |
| **P(HMM>B&H)** | block-bootstrap probability (1000 iters, 60-day blocks) that HMM Sharpe ≥ B&H Sharpe. |
| **Δ Sharpe** | HMM Sharpe minus B&H Sharpe. Positive = HMM edge. |
| **HMM** | canonical config: Gaussian HMM (K=3) + per-state vol-target chosen by walk-forward brute force. |
| **Walk-forward** | honest test: at each window, refit on data strictly before the test slice. No future info leaks. |
| **Block bootstrap** | confidence intervals via resampling blocks (default 60 days) instead of independent observations. |
| **Vol-target** | position size that holds expected portfolio vol constant. `pos = clip(target / pred_vol, 0, 1)`. |
| **Per-state target** | vol-target picked per HMM state. State 0 might get `target=0.10` (defensive), state 2 `target=0.40` (aggressive). |

## Architecture

```
   ┌────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
   │  Load OHLC │ →  │ Feature     │ →  │ Train RF     │ →  │ WF Training  │
   │  bars      │    │ Engineering │    │ vol forecast │    │ + Brute-Force│
   └────────────┘    └─────────────┘    └──────────────┘    │ Per-State    │
                                                                   │ Targets      │
                                                                   ▼
   ┌─────────────┐   ┌──────────────────┐   ┌────────────────┐
   │  Bootstrap   │ ← │  Position Sizing  │ ← │  Walk-Forward  │
   │  CIs + plots │   │  + TC             │   │  HMM K=3       │
   └─────────────┘   └──────────────────┘   └────────────────┘
```

### Position sizing (per bar)
```
pred_vol(t+1)  = RF prediction of next-bar realized vol
state_k(t)     = HMM posterior probability of regime k
target_k       = per-state vol-target (chosen in WF training by joint brute-force)
soft_pos_k(t)  = clip(target_k / pred_vol(t+1), 0, 1)
position(t)    = Σ_k state_k(t) · soft_pos_k(t)   # convex mixture
```

### What each component does
| Component | Learns | Fit cadence |
|---|---|---|
| Random Forest | next-period RV from 17 lagged features | once per asset |
| Gaussian HMM (K=3) | regime probabilities (calm / neutral / stress) | once per WF window |
| Per-state target | vol-target per regime | once per WF window |

## Project layout

```
.
├── README.md
├── LICENSE                         # MIT
├── requirements.txt
├── pyproject.toml
├── Makefile                        # make install / test / run-* / clean
├── .gitignore
│
├── config/
│   └── default.yaml                # all knobs in one place
│
├── data/                           # input OHLC bars (gitignored, README.md only)
│   ├── crypto_BTC_USD_1m*.parquet
│   ├── crypto_ETH_USD_1m*.parquet
│   ├── *_usd_1h.csv                # altcoins + stablecoins
│   └── cache/                      # LSE API response cache
│
├── src/
│   ├── multi_asset_runner.py       # crypto pipeline (hourly, 10 assets)
│   ├── equities_runner.py         # equities pipeline (daily via LSE)
│   ├── risk_strategy/              # extensible strategies
│   ├── diagnostics/                # residual scatter, Q-Q, year breakdowns
│   ├── data_io/                    # asset loaders (parquet + CSV + LSE)
│   └── utils/                      # metrics, config loader
│
├── scripts/
│   └── run_pipeline.py             # CLI entrypoint with progress + ETA
│
├── outputs/                        # generated plots + CSVs (gitignored)
│
├── docs/
│   ├── architecture.md             # pipeline diagram + responsibilities
│   ├── README-results.md           # full results report
│   ├── equities-results.md         # US small-cap equities headline
│   ├── russell2000-results.md      # Russell 2000 top-200 headline
│   ├── lse_api_notes.md            # London Strategic Edge API notes
│   ├── ux_review.md                # dummy-user UX review
│   └── adding_new_assets.md        # extension guide
│
├── tests/                          # 11 unit tests
└── .bak/                           # legacy / superseded scripts
```

## Reproducing the results

```bash
# Crypto multi-asset (80/20 split per asset, ~5 WF windows each)
python scripts/run_pipeline.py --assets BTC,ETH,ADA,BNB,DOGE,LINK,LTC,SOL,XRP

# US small-cap equities (daily bars via LSE)
python src/equities_runner.py

# Russell 2000 top-200 (provided CSV)
python src/equities_runner.py --from-csv data/russell2000_top200.csv \
    --output-dir outputs/russell2000_top200 --n-boot 500
```

Per-asset PNG plots + per-window CSVs + summary JSON land in `outputs/<ASSET>/`.

## Known limitations

1. **BTC only marginal edge** — BTC's mean-reverting vol + 24/7 liquidity leaves less room for HMM to add value vs Buy-and-Hold.
2. **XRP underperforms with default config** — see `docs/russell2000-results.md`.
   XRP's vol-of-vol is structurally higher (33% above BTC).
3. **Single-asset test folds vary in length** — bootstrap CIs are wider for short-history assets.
4. **Block-bootstrap on 60-day blocks gives ~30 effective independent samples**
   for a 5-year test fold. CIs are honest but conservative.
5. **No external data integration** — only OHLC. Sentiment / order-book / on-chain
   signals would likely help. The LSE API integration (see `docs/lse_api_notes.md`)
   is the price-tape layer.

## Tests

```bash
pytest tests/ -v           # 11 unit tests (metrics, position sizing, lookahead audit)
make test                  # same, via Makefile
```

## References

- [London Strategic Edge API docs](https://londonstrategicedge.com/api-documentation/)
- [hmmlearn](https://hmmlearn.readthedocs.io/) — Gaussian HMM implementation
- [scikit-learn RandomForestRegressor](https://scikit-learn.org/)

## License

MIT — see `LICENSE`.
