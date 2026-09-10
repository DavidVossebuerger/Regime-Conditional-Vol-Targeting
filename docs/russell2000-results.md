# Russell 2000 Top-200 Results

Generated from
`python src/equities_runner.py --from-csv data/russell2000_top200.csv --output-dir outputs/russell2000_top200 --n-boot 500`.

## Setup

- **Universe:** top 200 Russell 2000 components by index weight (Sept 2026 holdings)
- **Data:** daily bars via London Strategic Edge API, 500 iterations × 60-day block bootstrap
- **Pipeline:** HMM wc-feat (same as crypto/equities baseline), adaptive walk-forward
- **Coverage:** 177 of 200 symbols had ≥800 daily bars; 23 skipped (insufficient data or 404 on LSE)

## Headline — 78% beat Buy-and-Hold

| Metric | Value |
|---|---|
| Symbols processed | 177 |
| HMM beats B&H (Δ Sharpe > 0) | **138 (78%)** |
| HMM loses to B&H (Δ Sharpe < 0) | 39 (22%) |
| **Equal-weighted mean Δ Sharpe** | **+0.183** |
| Index-weighted Δ Sharpe | +0.180 |
| Median Δ Sharpe | +0.156 |
| Mean DD reduction | 0.175 |
| Mean B&H Sharpe → HMM Sharpe | +0.452 → **+0.634** |

The index-weighted Δ Sharpe essentially equals the equal-weighted one — meaning
the result is robust across small caps of different sizes, not just driven by
large winners like COMP or KRYS.

## Δ Sharpe distribution

```
count  177
mean   +0.183
std     0.401
min    -1.235
25%    +0.015
50%    +0.156
75%    +0.373
max    +1.521
```

3 out of 4 symbols have Δ Sharpe > 0. The bottom-quartile is barely negative
(median loser has Δ ≈ -0.05). A handful of outliers on the downside pull the
mean loser number down.

## Top 15 winners (by Δ Sharpe)

| Symbol | B&H Sharpe | HMM Sharpe | Δ Sharpe | P(HMM>B&H) | Max DD B&H → HMM |
|---|---|---|---|---|---|
| **COMP** (Compass Pathways) | 0.57 | 2.09 | **+1.52** | 0.69 | -44% → -14% |
| **HIMS** (Hims & Hers Health) | -0.05 | 1.28 | +1.32 | **0.98** | -73% → -26% |
| **MARA** (Marathon Digital) | -0.00 | 1.30 | +1.30 | 0.89 | -123% → -52% |
| **RGTI** (Rigetti Computing) | -0.28 | 0.85 | +1.13 | 0.61 | -65% → -13% |
| **ALKS** (Alkermes) | 0.34 | 1.40 | +1.06 | 0.91 | -41% → -25% |
| **VSAT** (ViaSat) | 0.05 | 0.95 | +0.90 | 0.93 | -94% → -33% |
| **MXL** (MaxLinear) | 0.88 | 1.74 | +0.86 | 0.91 | -101% → -33% |
| **KRYS** (Krystal Biotech) | 1.90 | 2.74 | +0.85 | 0.80 | -19% → -11% |
| **MIRM** (Mirum Pharma) | 0.73 | 1.56 | +0.83 | 0.70 | -41% → -21% |
| **CNR** (Cornerstone Build) | 0.35 | 1.14 | +0.79 | 0.87 | -38% → -18% |
| **HCC** (Warrior Met Coal) | 0.94 | 1.72 | +0.78 | 0.74 | -35% → -12% |
| **SSRM** (SSR Mining) | 1.10 | 1.88 | +0.78 | 0.82 | -38% → -20% |
| **EBC** (Eastern Bankshares) | 0.77 | 1.53 | +0.76 | 0.81 | -17% → -13% |
| **EAT** (Brinker Intl) | -0.33 | 0.43 | +0.75 | 0.87 | -126% → -64% |
| **LGND** (Ligand Pharma) | 1.18 | 1.90 | +0.72 | 0.80 | -32% → -13% |

**Pattern:** Healthcare/biotech names (COMP, HIMS, RGTI, ALKS, MIRM, KRYS) and small-cap
speculative names (MARA, VSAT) dominate the winner list. These tend to have
binary, jumpy return profiles that vol-targeting is well-suited to capture.

## Top winners by INDEX WEIGHT (most representative for the index)

| Symbol | Index Weight | Δ Sharpe | P(HMM>B&H) |
|---|---|---|---|
| **COMP** | 0.249% | +1.52 | 0.69 |
| **ALKS** | 0.247% | +1.06 | 0.91 |
| **VSAT** | 0.298% | +0.90 | 0.93 |
| **KRYS** | 0.295% | +0.85 | 0.80 |
| **SSRM** | 0.270% | +0.78 | 0.82 |
| **EAT** | 0.304% | +0.75 | 0.87 |
| **FROG** | 0.309% | +0.62 | 0.59 |
| **SNEX** | 0.246% | +0.61 | 0.63 |
| **PLXS** | 0.218% | +0.60 | 0.87 |
| **VLY** | 0.223% | +0.60 | 0.80 |

These 10 components alone cover ~2.7% of the Russell 2000 index weight and all
benefit meaningfully from HMM wc-feat.

## Worst 5 (where the model underperforms)

| Symbol | B&H Sharpe | HMM Sharpe | Δ Sharpe | P(HMM>B&H) | Likely cause |
|---|---|---|---|---|---|
| **OSCR** | +2.44 | +1.21 | **-1.24** | 0.32 | Strong uptrend with low vol; vol-targeting dragged returns |
| **NE** (Noble Corp) | -0.64 | -1.81 | -1.18 | 0.39 | Sustained downtrend with vol-spikes that reverted |
| **TWST** (Twist Bioscience) | +2.04 | +0.91 | -1.13 | 0.22 | Strong uptrend; vol-targeting underperformed |
| **SRRK** (Scholar Rock) | +0.62 | -0.19 | -0.80 | 0.09 | Loss period dominated the test fold |
| **PRAX** (Praxis Precision) | +0.05 | -0.69 | -0.75 | 0.07 | Loss period dominated |

**Pattern for losers:** OSCR and TWST both had very strong uptrends during the
test fold. Vol-targeting mathematically reduces exposure during quiet periods,
which costs returns. **For an investor, this is the cost of risk management** —
the same mechanism that protects against drawdowns also caps upside in trending
markets.

## Comparison to other asset classes

| Universe | N tested | N beat B&H | Mean Δ Sharpe | Mean DD cut |
|---|---|---|---|---|
| Crypto (10 assets) | 10 | 8 (80%) | +0.95 | -52% |
| US small-caps (manual, 9) | 9 | 7 (78%) | +0.93 | -47% |
| **Russell 2000 top-200 (this run)** | **177** | **138 (78%)** | **+0.18** | **-17%** |

The hit rate is **identical at 78%**, but the magnitude of improvement is
smaller in the Russell 2000 run. Two reasons:

1. **Heterogeneous dynamics** — the 9-asset manual basket was hand-picked for
   high-vol / meme / crypto-correlated names, where vol-targeting shines. The
   Russell 2000 has 200 components spanning many sectors and vol regimes,
   including stable mid-caps and slow growers where vol-targeting is a drag.
2. **Mostly positive B&H** — many Russell 2000 stocks had positive Buy-and-Hold
   Sharpe over the test window. Vol-targeting reduces positions in good times,
   dragging down performance for these names. (Same effect as the static-soft
   configuration: it never *loses* much, but it caps the upside.)

**Bottom line:** The risk-management system is robust on a broad small-cap
universe. Hit rate is consistent. Magnitude is asset-class dependent — best on
high-vol/jumpy names, less impactful on stable/treending names.

## How to reproduce

```bash
# 1. Set your LSE key (one-time)
cp .env.example .env
# edit .env to set LSE_API_KEY=...

# 2. Re-run top-200 (cached bars won't refetch)
python src/equities_runner.py --from-csv data/russell2000_top200.csv \
    --output-dir outputs/russell2000_top200 --n-boot 500

# 3. Or run your own basket
python src/equities_runner.py --assets AAPL,MSFT,GOOG --n-boot 1000
```

## Files

- `outputs/russell2000_top200/<SYMBOL>/equity.png` — per-asset equity / vol / DD
- `outputs/russell2000_top200/<SYMBOL>/per_window.csv` + `yearly.csv` + `summary.json`
- `outputs/russell2000_top200/summary.csv` — full 177-row metrics table
- `outputs/russell2000_top200/cross_asset.png` — Sharpe distribution across all 177 symbols
- `data/russell2000_top200.csv` — top-200 ticker list with weights
- `data/cache/lse_*.csv` — cached LSE responses (reused on rerun)
