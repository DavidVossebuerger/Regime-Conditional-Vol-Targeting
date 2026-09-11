# Changelog

## [1.1.0] — Unreleased

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
- Crypto pipeline (`src/multi_asset_runner.py`) — 10 assets, hourly bars
- Equities pipeline (`src/equities_runner.py`) — daily bars via LSE
- 11 unit tests: metrics, position sizing, lookahead audit (all passing)
- GitHub Actions CI: pytest on Python 3.11/3.12 + smoke pipeline
- Issue templates, MIT license, badges
- `Makefile`, `pyproject.toml`

### Notes
- Initial release used the worst-case feature; removed in 1.1.0 after validation showed no benefit.
- Russell 2000 top-200 result: 138/177 (78%) beat B&H with the original (more complex) pipeline.
- Per-asset timing: ~1–3s cold, ~0.5–1s warm with LSE cache.
