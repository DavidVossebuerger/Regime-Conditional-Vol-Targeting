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

### Caveats

- **SOL test fold is recent and concentrated.** The headline +2.53 ΔSharpe for
  SOL is heavily weighted by the most recent walk-forward window
  (2026-06-18 → 2026-09-01, RCVT Sharpe 6.73 on that window). The first two
  windows of the same test fold (Dec-2025 → Jun-2026) show RCVT Sharpes of
  0.54 and 0.06 respectively. Bootstrap mean (1.99) is robust across
  resamples (P=0.88 HMM>B&H), but the headline Sharpe is **not** representative
  of a "typical" SOL period — it reflects strong outperformance in the most
  recent ~10-week slice. Treat SOL as the asset with the highest *current*
  edge, not the highest *expected* edge.
- **BTC marginal, XRP negative.** BTC's mean-reverting vol + 24/7 liquidity
  leaves little room for vol-targeting to add value. XRP's vol-of-vol is
  structurally elevated; vol-targeting cuts exposure in the moves that pay
  off and adds it back in the chop.

### Per-regime decomposition (all 9 crypto assets)

The headline Sharpe gains do not come uniformly from all HMM regimes — see
[`docs/regime_decomposition.md`](regime_decomposition.md) for the full
breakdown. Summary across 9 assets:

- **SOL** stands out as the only asset with edge **distributed** across
  regimes (concentration 0.50, split between state 0 and state 1).
- **BNB** is the second-most-robust (0.55), and uniquely has positive
  edge in the **stress** regime (state 2).
- **DOGE + ETH** are 100 % concentrated in single regimes — fragile to
  regime shifts.
- **State 1 (chop)** is the dominant edge regime for 5 of 9 assets.
- **State 2 (stress)** is mostly neutral or negative — RCVT protects
  drawdowns but doesn't generate alpha in panics.
- **XRP** underperforms B&H; the loss is concentrated in state 2.

## Russell 2000 top-200 (daily via LSE, realistic TC = 10 bps/side)

178/200 in CSV (177 with valid data). **115/177 beat B&H (65%)**, mean Δ Sharpe +0.145, median +0.094.

The previous run used 2 bps/side which underestimates small-cap spread costs. This
run uses **10 bps/side = 20 bps round-trip** (auto-selected by `make run-russell`
because the filename contains "russell") — much closer to real small-cap
spreads (5-20 bps per side).

### Top 5 winners (realistic TC = 10 bps/side)

| Symbol | B&H Sharpe | HMM Sharpe | Δ Sharpe | P(HMM>B&H) |
|---|---|---|---|---|
| **CIFR** (Cipher Mining) | 0.08 | 1.88 | **+1.80** | 0.94 |
| **HCC** (Warrior Met Coal) | 0.91 | 2.15 | +1.24 | 0.85 |
| **HIMS** (Hims & Hers) | -0.12 | 1.01 | +1.13 | 0.81 |
| **VSAT** (ViaSat) | 0.05 | 1.16 | +1.11 | 0.94 |
| **CRSP** (CRISPR Therapeutics) | -0.33 | 0.77 | +1.10 | 0.91 |

### Bottom 5 (where HMM underperforms)

| Symbol | B&H Sharpe | HMM Sharpe | Δ Sharpe | Likely cause |
|---|---|---|---|---|
| **COGT** (Cogent Biosciences) | +1.06 | -0.23 | -1.29 | Strong uptrend; vol-targeting dragged |
| **OSCR** (Oscar Health) | +2.41 | +1.26 | -1.15 | Strong uptrend (consistent with 2bps run) |
| **LUMN** (Lumen) | -0.49 | -1.30 | -0.81 | Persistent downtrend |
| **PI** (Impinj) | +0.40 | -0.26 | -0.67 | Modest uptrend |
| **FLR** (Fluor) | +0.75 | +0.17 | -0.58 | Modest uptrend |

**Same pattern:** strong Buy-and-Hold uptrends with low vol → vol-targeting drags returns.

### Top 5 winners (optimistic TC = 2 bps/side — historical)

> This block is the historical run with TC = 2 bps/side (unrealistic for
> small caps). Kept here for comparison. For current canonical results use
> the 10 bps/side block above.

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

## Transaction cost impact

The pipeline subtracts `|Δpos(t)| · TC_PER_SIDE` from strategy returns at every
bar where position changes. TC is now asset-class-specific:

| Asset class | TC_PER_SIDE | Round-trip | Auto-selected by |
|---|---|---|---|
| Crypto | 2 bps | 4 bps | default |
| US large-cap equity | 2 bps | 4 bps | hand-picked basket |
| Russell 2000 small-cap | **10 bps** | **20 bps** | filename contains "russell" |

The 10 bps for small caps is based on realistic spreads (5-20 bps per side
common in Russell 2000 names). The earlier 2 bps was an over-optimistic
assumption that under-counted real execution costs.

**Effect on results** (Russell 2000):
- With 2 bps/side (too optimistic): 147/177 (83%) beat B&H, mean Δ Sharpe +0.25
- With 10 bps/side (realistic): 115/177 (65%) beat B&H, mean Δ Sharpe +0.15
- The hit-rate drops but **majority still beat B&H** — the risk-management
  edge is real but more modest than the optimistic run suggested.

TC is deducted at **two** stages to prevent over-fitting on round-trip costs:
1. **Walk-forward target optimization**: per-state targets are picked to maximize
   Sharpe *after* deducting TC on training data
2. **Live strategy returns**: TC subtracted from P&L for headline metrics

`tests/test_tc_deduction.py` verifies the deduction math directly (7 tests,
all passing).

## Cross-asset summary

| Universe | N tested | N beat B&H | Mean Δ Sharpe |
|---|---|---|---|
| Crypto (9 assets, hourly, TC=2bps) | 9 | 8 (89%) | **+1.26** |
| Russell 2000 top-200 (daily, TC=10bps) | 177 | 115 (65%) | **+0.15** |

The simpler pipeline (RF + HMM + per-state target) works across asset classes:
- Hit rate: **65–89%** (vs 78–80% previously with wc_feat)
- Mean Δ Sharpe: **+0.15 to +1.26**

The Russell 2000 hit-rate drops from 83% → 65% when TC is realistic (10 bps/side
instead of 2 bps/side). Still a clear majority beat B&H, but the magnitude
of the edge shrinks substantially. The HMM-Drawdown protection is the more
robust benefit — even on losers, Max-DD is generally better than B&H.

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
