# Crypto Risk-Management Pipeline

A regime-aware risk-management system for crypto portfolios. Combines a
**Random Forest vol forecast** with a **Gaussian HMM regime classifier** and a
**thermodynamic worst-case-loss feature** (Feng 2019, [arXiv:1904.00151](https://arxiv.org/abs/1904.00151))
to produce a soft-vol-targeted position sizing rule that **outperforms
Buy-and-Hold on 8 of 10 tested crypto assets** with strict walk-forward
validation and block-bootstrap significance.

> **Headline result (BTC deep run, 2020-01 → 2026-09, 24 walk-forward windows,
> 180-day test / 180-day step):** HMM wc-feat achieves **Sharpe 0.853** vs
> Buy-and-Hold's **0.550**, with Max-DD reduced from **-148% to -69%** and
> **95% vol-budget compliance**. Bootstrap 95% CI for HMM Sharpe is the only
> one with a strictly positive lower bound.
>
> **Multi-asset run (80/20 split per asset, 5 windows of 90 days each):**
> HMM wc-feat beats Buy-and-Hold on **8 of 10** assets; only **XRP** and
> the **stablecoin USDC** underperform (USDC has no volatility to target).

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify environment (smoke-test the critical imports)
python -c "import numpy, pandas, sklearn, hmmlearn, matplotlib; print('ok')"

# 3. (Optional) Live data via London Strategic Edge API (crypto + US equities)
#    Your LSE key can live in any of: ~/.crypto_risk_pipeline.env, .env, env var.
cp .env.example .env
# edit .env and set LSE_API_KEY=...
# See docs/lse_api_notes.md for what the API offers.

# 4a. Crypto pipeline (hourly bars from local data/*.csv|parquet)
python scripts/run_pipeline.py --assets BTC,ETH,ADA,BNB,DOGE,LINK,LTC,SOL,XRP

# 4b. Equities pipeline (daily bars from LSE API)
python src/equities_runner.py --assets RIOT,MARA,DKNG,LCID,RIVN,PLTR,RKT,OPEN,SOFI

# 5. Run unit tests
pytest tests/ -v
```

## Glossary

| Term | Meaning |
|---|---|
| **Sharpe** | annualized mean return / annualized vol of returns. Risk-adjusted return. |
| **Sortino** | like Sharpe but only downside std in denominator. |
| **Max DD** | maximum peak-to-trough drawdown, expressed as a positive log-return number. Smaller is better. |
| **Calmar** | annualized return / Max DD. Reward per unit of worst-case loss. |
| **B&H** | Buy-and-Hold baseline — always fully invested, no transaction costs. |
| **P(HMM>B&H)** | block-bootstrap probability (default 1000 iterations, 1-week blocks) that HMM Sharpe ≥ B&H Sharpe. ≥0.5 means "HMM at least as good". |
| **Δ Sharpe** | HMM Sharpe minus B&H Sharpe. Positive = HMM edge. |
| **DD Reduction** | (B&H Max DD) − (HMM Max DD). Positive = HMM has smaller drawdown. |
| **HMM wc-feat** | the canonical config: HMM regime classifier with vol-targeting + worst-case feature. |
| **V_wc** | thermodynamic worst-case expected return, `η⁻¹ log( mean(exp(η r)) )`. Higher = worse expected outcome under model perturbation. See `docs/papers/1904.00151.pdf`. |
| **η (eta)** | entropic budget — controls how much "model uncertainty" we assume. Larger η → more conservative. |
| **Walk-forward** | honest test: at each window, refit / retune on data strictly before the test slice. No future info leaks. |
| **Block bootstrap** | confidence intervals via resampling blocks (default 168h = 1 week) instead of individual observations, which would understate uncertainty. |
| **Vol-target** | position size that holds expected portfolio vol constant. `pos = clip(target / pred_vol, 0, 1)`. |

## Project layout

```
.
├── README.md                       # this file
├── LICENSE                         # MIT
├── requirements.txt
├── .gitignore
│
├── config/
│   └── default.yaml                # all knobs in one place
│
├── data/                           # input OHLC bars (CSV / parquet, hourly)
│   ├── crypto_BTC_USD_1m*.parquet
│   ├── crypto_ETH_USD_1m*.parquet
│   ├── *_usd_1h.csv                # ADA, BNB, DOGE, LINK, LTC, SHIB, SOL, XRP, …
│   └── …stablecoins/               # USDC, FDUSD (sanity check assets)
│
├── src/
│   ├── __init__.py
│   ├── multi_asset_runner.py       # entry point: walk-forward + bootstrap
│   ├── risk_pipeline_worstcase.py  # canonical BTC pipeline incl. V_wc
│   ├── risk_pipeline_hourly.py     # BTC-only baseline without worstcase
│   ├── risk_strategy/              # HMM, soft-weight, vol-targeting strategies
│   ├── diagnostics/                # residual scatter, Q-Q, year breakdowns
│   ├── data_io/                    # asset loaders (parquet + CSV)
│   └── utils/                      # metrics, config loader, progress bar
│
├── scripts/
│   └── run_pipeline.py             # CLI entrypoint with rich progress + ETA
│
├── outputs/                        # generated plots + CSVs (gitignored)
│   ├── btc_hourly/
│   ├── btc_worstcase/
│   └── multi_asset/
│
├── docs/
│   ├── README-results.md           # full results report (latest run)
│   ├── xrp_failure_analysis.md     # why XRP underperforms (generated)
│   ├── lse_api_notes.md            # London Strategic Edge API notes
│   └── papers/
│       └── 1904.00151.pdf          # Feng 2019 — thermodynamic worst-case
│
└── .bak/                           # legacy / superseded scripts
    ├── scripts_legacy/             # earlier iteration of HMM strategies
    ├── lib_legacy/                 # XAU-only VolatilityBacktester class
    ├── results_legacy/             # outputs from earlier exploration
    ├── legacy_xau_daily/           # original XAU volatility project
    ├── legacy_risiko_optimierung/  # original Risiko-Optimierung folder
    ├── repo_prune_backup/          # pre-prune backup of repo
    ├── btc_csv_duplicates/         # redundant 1m CSVs (kept, parquets are canonical)
    └── *.zip                       # deployment zips
```

## Architecture

### Pipeline (per asset)

```
   ┌────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
   │  Load OHLC │ →  │ Feature     │ →  │ Train RF     │ →  │ Calibrate η  │
   │  1h bars   │    │ Engineering │    │ vol forecast │    │ on snippet   │
   └────────────┘    └─────────────┘    └──────────────┘    └──────┬───────┘
                                                                   │
                                                                   ▼
   ┌─────────────┐   ┌──────────────────┐   ┌────────────────┐   ┌─────────┐
   │  Bootstrap   │ ← │  Build V_wc       │ ← │  Walk-Forward  │ ← │ Pick η  │
   │  CIs + plots │   │  rolling 168h     │   │  HMM K=3       │   │         │
   └─────────────┘   └──────────────────┘   └────────────────┘   └─────────┘
```

### Position sizing (per hour)

```
pred_vol(t+1)         = RF prediction of next-bar realized vol
V_wc(t, η=4.0)        = worst-case expected return under entropic budget
                        (Cramér-Lundberg dual of exp(η r))
state_k(t)            = HMM posterior probability of regime k
target_k              = per-state vol-target chosen by walk-forward training

soft_pos_k(t)         = clip(target_k / pred_vol(t+1), 0, 1)
position(t)           = Σ_k state_k(t) · soft_pos_k(t)        # convex mixture
```

Multiplicative risk-scaler (`1 / V_wc`) is available but **not used in the
canonical config** — empirically it over-corrects on top of the HMM vol-target
(see `docs/xrp_failure_analysis.md`).

### Why the HMM + worst-case combo works

- **Random Forest** captures the **vol-level** (linear in lagged RV features).
- **HMM** captures the **regime** (high-vol-persistent vs low-vol-reverting)
  using rolling vol-of-vol and z-scores.
- **Worst-case feature** (Cramér-Lundberg dual) captures the **tail-risk**
  separately from the vol-level. This is what distinguishes "low vol because
  the market is calm" from "low vol because everyone is hedging" — the latter
  has heavier tails.

## Reproducing the results

There are three valid configurations. Pick based on what you want to test:

### Run 1: BTC deep walk-forward (24 windows over 6 years)

Uses a **fixed date cutoff** (`config/default.yaml` → `data.test_start_date = "2020-01-01"`)
so the test fold spans the full 2020 → 2026 crypto cycle (COVID, bull, crash, ETF, ATH).
This is the headline-run config that produces **BTC Sharpe 0.853 vs 0.550**.

```bash
# Edit config/default.yaml: set data.test_start_date: "2020-01-01"
python scripts/run_pipeline.py --assets BTC --n-boot 2000
```

### Run 2: Crypto multi-asset comparison (5 windows of 90 days each)

Uses **80/20 split per asset**, so each asset gets a test fold proportional to its
history length. Useful for cross-asset comparison but each individual asset has
fewer walk-forward windows → wider bootstrap CIs.

```bash
# Leave data.test_start_date: null in config/default.yaml
python scripts/run_pipeline.py --assets BTC,ETH,ADA,BNB,DOGE,LINK,LTC,SOL,XRP
```

### Run 3: US small-cap equities (daily bars via LSE API)

Same HMM wc-feat pipeline, adapted for daily bars (`periods_per_year=252`).
The LSE loader (`src/data_io/lse_loader.py`) fetches bars and caches them in
`data/cache/`. **7 of 9 tested small caps beat Buy-and-Hold.** See
`docs/equities-results.md`.

```bash
# 1) Set your key (one-time)
cp .env.example .env  # then edit .env to set LSE_API_KEY=...
# 2) Run
python src/equities_runner.py
```

### Multi-asset expected output (Run 2)

Approximate, due to RF + HMM stochasticity:

| Asset | B&H Sharpe | HMM Sharpe | Max DD B&H | Max DD HMM |
|---|---|---|---|---|
| SOL  | -0.51 | +2.55 | -89% | -22% |
| DOGE | -1.58 | +0.26 | -137% | -49% |
| ETH  | -0.06 | +1.60 | -118% | -38% |
| LTC  | -0.72 | +0.90 | -122% | -43% |
| ADA  | -1.18 | +0.22 | -199% | -62% |
| BNB  | +0.11 | +1.46 | -93% | -27% |
| LINK | -1.07 | +0.14 | -130% | -39% |
| BTC  | -0.61 | -0.42 | -77% | -56% |
| XRP  | -0.67 | -0.85 | -131% | -88% |
| USDC | +0.11 | +0.11 | -0.4% | -0.4% |

## Known limitations

1. **XRP underperforms** with default config — see `docs/xrp_failure_analysis.md`.
   The system isn't broken, it's mis-tuned for XRP:
   - **Tier-1 fix:** `η=4.0` (instead of 1.0) + `K=4` (instead of 3) lifts Sharpe
     from −0.85 → −0.01 (parity with BTC tier). Add XRP-ETH rolling correlation
     as a 5th HMM feature → Sharpe −0.85 → −0.57, Max-DD −88% → −75%.
   - **Structural cause:** XRP's vol-of-vol is 33% higher than BTC (median 0.67 vs 0.50,
     p95 1.91 vs 1.33). Jump-driven idiosyncratic news shocks (SEC rulings,
     delistings) make regime transitions harder to resolve with default K=3.
   - For production: drop XRP from the HMM vol-targeting sleeve unless
     Tier-1 fixes are applied.
2. **BTC only marginal edge** — BTC's mean-reverting vol structure + 24/7
   liquidity leaves less room for the HMM to add edge vs Buy-and-Hold.
   ~Sharpe +0.19 in the multi-asset run is real but small.
3. **Single-snippet η calibration** — picked randomly from train fold.
   Multi-snippet averaging would be more robust.
4. **No external data integration** — only OHLC. Sentiment / order-book /
   on-chain signals would likely help. The LSE API integration (see
   `docs/lse_api_notes.md`) is the price-tape layer; sentiment/on-chain
   would need additional vendors.
5. **Bootstrap CI wide** — block-bootstrap on 168h blocks gives ~30 effective
   independent samples for a 5-year test fold. Confidence intervals are honest
   but conservative.
2. **BTC only marginal** — BTC's mean-reverting vol structure + 24/7 liquidity leaves
   less room for the HMM to add edge vs Buy-and-Hold. ~Sharpe +0.2 is real but small.
3. **Single-snippet η calibration** — picked randomly from train fold. Multi-snippet
   averaging would be more robust.
4. **No external data integration** — only OHLC. Sentiment / order-book / on-chain
   signals would likely help (LSE integration is the price-tape layer).
5. **Bootstrap CI wide** — block-bootstrap on 168h blocks gives ~30 effective
   independent samples for 5.8-year test fold. Confidence intervals are honest
   but conservative.

## Tests

```bash
pytest tests/                  # unit tests for metrics, position-sizing, V_wc
```

## References

- Feng, Y. (2019). *A Thermodynamic Picture of Financial Market and Model Risk*.
  arXiv:1904.00151. Used for the worst-case expected return feature.
- Hansen, L.P. & Sargent, T.J. (2008). *Robustness*. Princeton University Press.
  Foundation for the entropic budget interpretation.
- London Strategic Edge API docs: https://londonstrategicedge.com/api-documentation/

## License

MIT — see `LICENSE`.
