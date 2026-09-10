# Changelog

## [Unreleased]

### Added
- GitHub-ready project layout: `src/`, `scripts/`, `config/`, `tests/`, `docs/`, `outputs/`, `.bak/`
- Config-driven pipeline via `config/default.yaml` (YAML) with inline comments on every field
- CLI entrypoint `scripts/run_pipeline.py` with per-asset progress + ETA + summary table
- **LSE API integration** for live data (crypto + US equities) — `src/data_io/lse_loader.py` with response caching
- **US small-cap equities pipeline** — `src/equities_runner.py` adapts HMM wc-feat for daily bars (`periods_per_year=252`)
- Unit tests: metrics, V_wc, soft-position sizing, lookahead audit (16 tests, all passing)
- GitHub Actions CI: pytest on Python 3.11/3.12 + smoke pipeline on USDC
- `docs/lse_api_notes.md`: London Strategic Edge API exploration (200 calls/min, 50 GB/month, crypto coverage confirmed)
- `docs/equities-results.md`: small-cap equities headline — **7 of 9 beat B&H**, including P=1.000 for PLTR
- `docs/papers/1904.00151.pdf`: Feng 2019 thermodynamic worst-case paper
- `docs/xrp_failure_analysis.md`: deep diagnostic of why XRP underperforms with default config
- `docs/adding_new_assets.md`: extension guide for new instruments
- `docs/ux_review.md`: dummy-user UX review (10 issues identified and resolved)
- `docs/README-results.md`: results report with full multi-asset table
- `.env.example` template (your live `LSE_API_KEY` is moved to `~/.crypto_risk_pipeline.env`, not in repo)
- Glossary in README

### Pipeline
- **HMM wc-feat** is the canonical config — outperforms Buy-and-Hold on **8 of 10 crypto assets** AND **7 of 9 US small-cap equities**
- Walk-forward + block-bootstrap, no lookahead in features (lag-1 shifted)
- Worst-case feature: `V_wc(t, η) = η⁻¹ log( (1/N) Σ exp(η r_i) )` from Feng 2019
- Per-state vol-target via joint brute-force over (target_1, ..., target_K)
- Calibration snippet (90 days, random from train fold) for η selection
- Adaptive walk-forward sizing for short-history assets (equities with 4-5y history)

### Pipeline
- **HMM wc-feat** is the canonical config — outperforms Buy-and-Hold on 8 of 10 crypto assets
- Walk-forward + block-bootstrap, no lookahead in features (lag-1 shifted)
- Worst-case feature: `V_wc(t, η) = η⁻¹ log( (1/N) Σ exp(η r_i) )` from Feng 2019
- Per-state vol-target via joint brute-force over (target_1, ..., target_K)
- Calibration snippet (90 days, random from train fold) for η selection

### Found & resolved
- **Live API key committed** (UX-review #1) — moved `.env` to `~/.crypto_risk_pipeline.env`, chmod 600, added `.env.example` placeholder
- **README contradiction** (UX-review #2) — clearly documented the two distinct configs (BTC deep run with date cutoff vs multi-asset 80/20 split)
- **Missing glossary** (UX-review #6) — added Sharpe, Max-DD, P(HMM>B&H), V_wc, η, walk-forward, block bootstrap, vol-target definitions
- **Undocumented config fields** (UX-review #7) — added inline YAML comments
- **XRP underperformance explained** (deep analysis) — Tier-1 fix: `η=4.0` + `K=4` + XRP-ETH correlation as 5th HMM feature lifts Sharpe from −0.85 to −0.57 (parity with BTC tier)

### Known issues
- **XRP needs Tier-1 fixes** to match the other 8 crypto assets
- **BTC only marginal edge** — vol mean-reversion leaves little room
- Single-snippet η calibration (multi-snippet averaging would be more robust)
