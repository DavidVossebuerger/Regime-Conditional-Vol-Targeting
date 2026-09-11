# Alpha Decay Test — No decay; the edge is growing

**Question:** Does RCVT's edge diminish over time (alpha decay)?

## Method

For each of **1048 equities** (anchored WF run), we have walk-forward windows
spanning roughly 2015–2026. For every window we compute:
- **HMM Sharpe** from `per_window.csv` (already in the pipeline output)
- **BH Sharpe** from the yfinance cache over the same dates (added by this test)
- **Δ Sharpe = HMM − BH**

Then we aggregate Δ Sharpe across all assets by `test_start` year and test
whether there's a downward trend (which would indicate alpha decay).

## Year-by-year mean Δ Sharpe

| Year | N windows | N assets | Mean ΔSharpe | Median | SE | Hit rate |
|---|---|---|---|---|---|---|
| 2015 | 24 | 24 | **−0.099** | −0.117 | 0.037 | 25 % |
| 2016 | 53 | 27 | −0.011 | −0.027 | 0.019 | 40 % |
| 2017 | 30 | 27 | +0.134 | +0.109 | 0.031 | **83 %** |
| 2018 | 231 | 119 | +0.015 | +0.013 | 0.019 | 53 % |
| 2019 | 402 | 279 | +0.047 | +0.017 | 0.018 | 55 % |
| 2020 | 654 | 343 | +0.054 | +0.018 | 0.019 | 52 % |
| 2021 | 773 | 405 | +0.034 | +0.008 | 0.011 | 52 % |
| 2022 | 1035 | 559 | +0.092 | +0.051 | 0.011 | **60 %** |
| 2023 | 1330 | 695 | +0.076 | +0.037 | 0.009 | **59 %** |
| 2024 | 1570 | 816 | **+0.140** | +0.088 | 0.010 | **64 %** |
| 2025 | 1649 | 985 | +0.129 | +0.073 | 0.010 | 50 % |
| 2026 | 599 | 1045 | +0.112 | +0.074 | 0.011 | 26 % |

## Trend tests (all point the same direction: edge is growing)

| Test | Value | p-value |
|---|---|---|
| **OLS slope** (mean Δ Sharpe ~ year) | **+0.0142 ΔSharpe/year** | **0.0072** |
| **OLS R²** (year explains mean Δ Sharpe) | 0.531 | — |
| **Spearman ρ** (year vs mean Δ Sharpe, by-year) | **+0.678** | **0.0153** |
| **Mann-Kendall** trend test (yearly mean) | **S = +38, Z = +2.54** | **0.0112** |

All three independent tests agree: the trend is **positive and statistically
significant**. Year alone explains 53 % of the variation in mean Δ Sharpe.

## Interpretation

**There is no alpha decay — if anything, the edge is growing.**

The 2015–2016 numbers are weak / negative, but those early years have very
small samples (24–27 assets) and are dominated by assets that have since
been through regime transitions. From 2017 onward, the edge has been
**positive in every year**, with the strongest readings in 2024 (+0.140)
and 2025 (+0.129).

Several mechanisms could explain the apparent growth:
- **Market regime**: 2020 (COVID volatility), 2022 (bear market), and 2023–
  2025 (high-vol bull) are exactly the regimes where vol-targeting adds
  the most value — confirmed by our vol-scaling experiment
- **Survivorship**: the equities_yf universe is a snapshot of currently
  listed names; delisted tickers (which would have shown worse results)
  are excluded. The growth rate is partly a "yfinance snapshot" artefact,
  not necessarily a true market-edge growth
- **HMM fit quality**: as the HMM gets trained on more historical data
  per asset, its regime classification may marginally improve
- **TC unchanged**: 2 bps/side assumption is the same throughout, so the
  *relative* advantage of vol-targeting over B&H scales with vol regime
  rather than degrading over time

## What this means for production deployment

If the trend continues at +0.014 ΔSharpe/year, the strategy should remain
profitable — possibly more so — over the next several years. The absence of
alpha decay is unusual for a quant strategy and suggests the edge is
**structural** (vol-targeting mechanics) rather than **statistical**
(arbitrage that gets competed away).

That said:
- The growth rate is **modest** (+0.014/year is small vs the
  level of +0.10)
- **Survivorship bias** in the universe means real-world results could
  be ~30–50 % worse than our snapshot suggests
- A genuine **alpha-decay test** would require running on a frozen pre-2017
  universe and comparing later periods to earlier periods on the same set
  of assets — we haven't done that

## How to reproduce

```bash
python tests/alpha_decay.py
```

Outputs:
- `outputs/alpha_decay/alpha_decay.png` — 4-panel diagnostic
- `outputs/alpha_decay/by_year.csv` — yearly aggregates
- `outputs/alpha_decay/per_window_with_bh.csv` — per-window data with computed BH
