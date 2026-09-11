# Results (canonical pipeline — RF vol forecast + HMM regime + per-state target)

Latest run after removing the Feng-2019 worst-case feature (see CHANGELOG.md 1.1.0).

## Crypto multi-asset (80/20 split per asset)

| Asset | B&H Sharpe | HMM Sharpe | Δ Sharpe | P(HMM>B&H) | Max DD B&H | Max DD HMM | DD cut |
|---|---|---|---|---|---|---|---|
| **SOL** | -0.51 | +2.02 | **+2.53** | 0.88 | -89% | -27% | 70% |
| **ADA** | -1.18 | +0.78 | +1.96 | 0.93 | -199% | -60% | 70% |
| **LTC** | -0.72 | +1.06 | +1.78 | 0.95 | -122% | -31% | 75% |
| **LINK** | -1.07 | +0.61 | +1.68 | 0.93 | -130% | -34% | 74% |
| **BNB** | +0.11 | +1.72 | +1.61 | 0.88 | -93% | -23% | 75% |
| **DOGE** | -1.58 | -0.21 | +1.36 | 0.86 | -137% | -72% | 47% |
| **ETH** | -0.06 | +0.81 | +0.87 | 0.75 | -118% | -39% | 67% |
| BTC | -0.61 | -0.58 | +0.03 | 0.52 | -77% | -69% | 10% |
| XRP | -0.67 | -1.17 | -0.50 | 0.39 | -131% | -90% | 31% |

**Summary:** 8/9 beat B&H (89%). Mean Δ Sharpe +1.258, Median +1.608.

## Russell 2000 top-200 (daily via LSE)

177/200 had valid data. **147/177 beat B&H (83%)**, mean Δ Sharpe +0.253, median +0.205.

### Top 5 winners

| Symbol | B&H Sharpe | HMM Sharpe | Δ Sharpe | P(HMM>B&H) |
|---|---|---|---|---|
| **CIFR** (Cipher Mining) | 0.07 | 2.56 | **+2.49** | **1.000** |
| **QBTS** (D-Wave Quantum) | -0.51 | 1.16 | +1.66 | 0.81 |
| **VSAT** (ViaSat) | 0.05 | 1.25 | +1.20 | 0.95 |
| **HCC** (Warrior Met Coal) | 0.94 | 2.09 | +1.15 | 0.82 |
| **ALKS** (Alkermes) | 0.34 | 1.48 | +1.14 | 0.92 |

### Bottom 5 (where HMM underperforms)

| Symbol | B&H Sharpe | HMM Sharpe | Δ Sharpe | Likely cause |
|---|---|---|---|---|
| **OSCR** (Oscar Health) | +2.44 | +0.50 | **-1.94** | Strong uptrend; vol-targeting dragged returns |
| **COGT** (Cogent Biosciences) | +1.13 | -0.20 | -1.32 | Strong uptrend |
| **FROG** (JFrog) | +0.83 | +0.03 | -0.80 | Modest uptrend |
| **LUMN** (Lumen) | -0.50 | -1.25 | -0.75 | Persistent downtrend |
| **XENE** (Xencor) | +0.93 | +0.42 | -0.50 | Mid-cap volatility |

**Pattern for losers:** Same as before — strong Buy-and-Hold uptrends with low vol. Vol-targeting mathematically reduces exposure in good times.

## Cross-asset summary

| Universe | N tested | N beat B&H | Mean Δ Sharpe |
|---|---|---|---|
| Crypto (9 assets, hourly) | 9 | 8 (89%) | **+1.26** |
| Russell 2000 top-200 (daily) | 177 | 147 (83%) | **+0.25** |

The simpler pipeline (RF + HMM + per-state target, no Feng worst-case) **works
better** than the previous more-complex one:
- Hit rate: **83–89%** (vs 78–80% previously with wc_feat)
- Mean Δ Sharpe: **+0.25 to +1.26** (vs +0.18 to +0.95 previously)

The removed worst-case feature was noise rather than signal.

## How to reproduce

```bash
# Crypto (80/20 split, ~9 assets, 5 WF windows each)
python src/multi_asset_runner.py --assets BTC,ETH,ADA,BNB,DOGE,LINK,LTC,SOL,XRP

# Russell 2000 (177 valid symbols, 4-5 WF windows each)
python src/equities_runner.py --from-csv data/russell2000_top200.csv \
    --output-dir outputs/russell2000_top200 --n-boot 500
```

## Files

- `outputs/multi_asset/{ASSET}/equity.png` — per-asset equity / vol / drawdown
- `outputs/multi_asset/per_asset_summary.csv` — full metrics table
- `outputs/multi_asset/cross_asset.png` — Sharpe / Max-DD comparison
- `outputs/russell2000_top200/{SYMBOL}/equity.png` — per-asset equity
- `outputs/russell2000_top200/summary.csv` — 177-row metrics table
- `outputs/russell2000_top200/cross_asset.png` — full Russell 2000 comparison
