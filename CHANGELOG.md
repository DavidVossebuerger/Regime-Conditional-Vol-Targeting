# Changelog

## [1.1.1] — Unreleased

### Changed
- **Consolidated `metrics()` and `sharpe_rank()` into `src/utils/metrics.py`** — removed local copies from `multi_asset_runner.py` and `equities_runner.py`. Single canonical implementation with `pct_within_max` and `vol_window` params (defaults 0.60/24 for crypto, 0.40/20 for equities).
- **README "Canonical config" section** is now honest: explicitly says there is no YAML config layer, defaults live as constants + CLI flag defaults.
- **`docs/results.md`** adds a SOL test-fold caveat: the headline +2.53 ΔSharpe is dominated by the 2026-06-18 → 2026-09-01 window (Sharpe 6.73); first two windows show 0.54 / 0.06.
- **`docs/vol_scaling_results.md`** adds statistical caveat: n=45 is from 9 assets × 5 correlated scales, effective N ≈ 5–9, not 45.

### Removed
- **`config/default.yaml`** and the `config/` directory — runners never loaded YAML; the file was lying documentation.
- **`--no-cache` flag** in `equities_runner.py` — caching removed in v1.1.0, flag was a dead no-op since.
- **`--n-boot` flag** in `scripts/run_pipeline.py` — runner doesn't accept it; replaced by `--tc-per-side` forwarding (which IS supported by the runner).
- **`run-crypto-full` Makefile target** — was identical to `run-crypto`.
- **Stale `results_smoke/multi_asset` default** in `multi_asset_runner.py` argparse — replaced with `outputs/multi_asset`.

## [1.1.0]

### Removed
- **Thermodynamic worst-case feature (Feng 2019)** and the entire `risk_pipeline_worstcase.py` script
  - Did not provide measurable benefit in the canonical HMM pipeline; added complexity without statistical justification
  - Removed from `multi_asset_runner.py`, `equities_runner.py`, `config/default.yaml`
  - Removed `tests/test_worstcase.py`, `docs/papers/1904.00151.pdf`, `docs/xrp_failure_analysis.md`
- **Legacy BTC scripts** (`risk_pipeline_hourly.py`, `risk_pipeline_worstcase.py`)
- **Deprecated `--no-wc` flag** in `scripts/run_pipeline.py`

### Changed
- **README** rewritten — simpler TL;DR, accurate architecture diagram (no `η`, no `V_wc`), updated glossary
- **CHANGELOG** rewritten to reflect current state

## [1.0.0] — Initial public release (2026-09-11)

### Added
- GitHub-ready project layout: `src/`, `scripts/`, `config/`, `tests/`, `docs/`, `outputs/`, `.bak/`
- Config-driven pipeline via `config/default.yaml` (YAML) with inline comments on every field
- CLI entrypoint `scripts/run_pipeline.py` with per-asset progress + ETA + summary table
- LSE API integration for live crypto + US equity data (`src/data_io/lse_loader.py`)
- Crypto pipeline (`src/multi_asset_runner.py`) — 9 assets, hourly bars
- Equities pipeline (`src/equities_runner.py`) — daily bars via LSE
- 18 unit tests: metrics, position sizing, lookahead audit, transaction-cost deduction (all passing)
- GitHub Actions CI: pytest on Python 3.11/3.12 + smoke pipeline
- Issue templates, MIT license, badges
- `Makefile`, `pyproject.toml`

### Notes
- Initial release used the worst-case feature; removed in 1.1.0 after validation showed no benefit.
- Russell 2000 top-200 result: 138/177 (78%) beat B&H with the original (more complex) pipeline.
- Per-asset timing: ~1–3s cold, ~0.5–1s warm with LSE cache.
