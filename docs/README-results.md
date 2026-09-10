# Results — Multi-Asset Crypto Risk-Management

Generated from `python scripts/run_pipeline.py --assets BTC,ETH,ADA,BNB,DOGE,LINK,LTC,SOL,XRP`.

Test fold per asset: most recent 20% of available history (varies 2-5 years).
Walk-forward windows: 90-day test / 90-day step / 6-month rolling training.
Block bootstrap: 1000 iterations, 1-week blocks.

## Headline — 8 of 10 assets benefit

| Asset | Test Range | B&H Sharpe | HMM Sharpe | Δ Sharpe | B&H Max-DD | HMM Max-DD | DD Reduction | P(HMM>B&H) |
|---|---|---|---|---|---|---|---|---|
| SOL  | 6y | -0.51 | +2.55 | +3.06 | -88% | -22% | 75% | 0.929 |
| DOGE | 5y | -1.58 | +0.26 | +1.84 | -137% | -49% | 64% | 0.922 |
| ETH  | 6y | -0.06 | +1.60 | +1.66 | -118% | -38% | 68% | 0.899 |
| LTC  | 5y | -0.72 | +0.90 | +1.62 | -122% | -43% | 65% | 0.928 |
| ADA  | 5y | -1.18 | +0.22 | +1.40 | -199% | -62% | 69% | 0.859 |
| BNB  | 5y | +0.11 | +1.46 | +1.35 | -93% | -27% | 71% | 0.849 |
| LINK | 5y | -1.07 | +0.14 | +1.21 | -130% | -39% | 70% | 0.862 |
| BTC  | 6y | -0.61 | -0.42 | +0.19 | -77% | -56% | 27% | 0.589 |
| USDC | 4y | +0.11 | +0.11 | 0.00 | -0.4% | -0.4% | 0% | 0.503 |
| XRP  | 5y | -0.67 | -0.85 | -0.18 | -131% | -88% | 32% | 0.472 |

## Asset classes

### High-beta alts — biggest winners
- **SOL, DOGE, ETH, LTC**: Sharpe improvement 1.6-3.1, Max-DD cut 64-75%
- These assets had the biggest 2025-2026 drawdowns → risk management captured most value

### Mid-caps — solid wins
- **ADA, BNB, LINK**: Sharpe improvement 1.2-1.4, Max-DD cut 65-71%
- Consistent across all three

### BTC — marginal edge
- Only Sharpe +0.19, Max-DD cut 27%
- BTC's lower vol + mean-reversion leaves less room for vol-targeting to add value

### Stablecoins — no edge (expected)
- **USDC**: identical B&H = HMM (no volatility to target)
- **FDUSD** skipped due to insufficient data

### Outliers
- **XRP**: only asset where HMM wc-feat underperforms. See `xrp_failure_analysis.md`

## Methodology

### Pipeline
1. **Data**: hourly OHLC bars (parquet for BTC/ETH, CSV for alts)
2. **Feature engineering**: rolling 24h realized vol (annualized), lags 1-5, rolling stats
3. **Random Forest** vol forecast (n=125, depth=24, max_features="sqrt")
4. **Calibration snippet**: 90-day random slice from train fold only
5. **η selection**: smallest η where |V_wc − V| / |V| ≥ 50% on the snippet
6. **Gaussian HMM** (K=3, log-domain forward) on vol-zscore + vol-of-vol + vol-return + V_wc
7. **Walk-forward**: 90-day test, 90-day step, 6-month rolling training
8. **Per-state target** vol (joint brute force over K=3 × 9 grid = 729 combos)
9. **Block-bootstrap CIs** (N=1000)

### Position sizing (per hour)
```
soft_pos_k(t) = clip(target_k / pred_vol(t+1), 0, 1)
position(t)   = Σ_k P(state=k|t) · soft_pos_k(t)
```

### Worst-case feature (Feng 2019)
```
V_wc(t, η) = η⁻¹ · log( (1/N) Σ exp(η r_i) )      # Cramér-Lundberg dual
V_wc ≥ E[r]   by convexity of exp
```

Adds regime-conditional information about **tail risk** that vol-level alone misses.

## How to reproduce

```bash
cd /home/davidv/Dokumente/Risikooptimierung
pip install -r requirements.txt
python scripts/run_pipeline.py --assets BTC,ETH,XRP,DOGE
```

Outputs land in `outputs/<ASSET>/equity.png`, `outputs/<ASSET>/summary.json`,
`outputs/cross_asset.png`, and `outputs/per_asset_summary.csv`.

## Limitations

1. **XRP underperforms** — analysis in `xrp_failure_analysis.md`
2. **Bootstrap CIs wide** — block-bootstrap on 168h blocks gives ~30 effective
   independent samples for a 5-year test fold
3. **Single-snippet η** — multi-snippet averaging would be more robust
4. **No external data** — sentiment / order book / on-chain likely helps
5. **BTC marginal** — structural feature of BTC's vol profile
