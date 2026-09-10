# UX / Onboarding Review — FX-Fair-Value / Risikooptimierung

Author: simulated first-time user, no quant-finance background.
Scope: README.md, config/default.yaml, docs/README-results.md, docs/lse_api_notes.md, .env, scripts/, src/, tests/.
Method: read docs end-to-end, then actually executed the Quick Start commands.

Verdict: the project runs out-of-the-box, but the docs are inconsistent, assume quant literacy, and expose a live API key in the repo.

---

## 1. README.md — Quick Start

### 1.1 Pipeline does run, but the README does not warn about the live `.env`
- File/section: `README.md` § "Quick start", `.env`
- Issue: I ran `python scripts/run_pipeline.py --assets USDC` from the project root. It executed without errors in ~5s. **But** the `.env` file contains a real, committed `LSE_KEY=lse_live_53ac...` (41-char production-tier key). Quick Start never mentions `.env` at all. New users have no idea this file exists, let alone that they're shipping a key.
- Suggestion: add a Quick Start step `cp .env.example .env` (after creating a stub `.env.example`), and rotate the committed key.

### 1.2 Quick Start does not mention a virtualenv
- File/section: `README.md` § "Quick start"
- Issue: `pip install -r requirements.txt` is the first step, with no recommendation to use a venv. Requirements pin `numpy>=1.26, hmmlearn>=0.3, arch>=6.0, ...` — `arch` in particular is heavy and easy to break system-wide. Most Python users now expect venv steps.
- Suggestion: add `python -m venv .venv && source .venv/bin/activate` as the first command.

### 1.3 Quick Start step 2 silently passes even if dependencies are broken
- File/section: `README.md` § "Quick start", line 23
- Issue: `python -c "import numpy, pandas, sklearn, hmmlearn, matplotlib; print('ok')"` does not include `arch`, `optuna`, `tqdm`, `rich`, `yaml` — all of which are required by the pipeline. So this smoke test would pass even if several real dependencies are missing.
- Suggestion: replace with `python -c "import arch, hmmlearn, optuna, rich, yaml; print('ok')"` (or just `pip install -r requirements.txt && python -c "from src.multi_asset_runner import main"`).

### 1.4 Expected-output table contradicts the Headline block
- File/section: `README.md`, vs § "Headline result" vs § "Expected output"
- Issue: The headline says BTC 1h gets "Sharpe 0.853 vs B&H 0.550, 24 walk-forward windows". The "Expected output" table 20 lines further down lists BTC as `B&H Sharpe = -0.61, HMM Sharpe = -0.42`. These are two completely different BTC results — different Sharpe, different sign, different number of windows. New users will think the docs are broken or one of them is from a different project.
- Suggestion: pick one source of truth and reconcile (or label both clearly, e.g. "headline = config A, table = config B").

### 1.5 The "8 of 10 assets" claim needs to be made literally visible in the table
- File/section: `README.md` § "Headline result"
- Issue: README opens with "outperforms Buy-and-Hold on 8 of 10 tested crypto assets". The expected-output table shows Sharpe values where BTC has *both* B&H and HMM negative. Strictly reading the table, BTC does NOT outperform B&H (both negative), and USDC ties. A careful reader counts 6 alts with a positive Sharpe-delta, not 8.
- Suggestion: add a `beats B&H?` column with the actual boolean (or a sort/colour) so the "8 of 10" claim is mechanically visible.

### 1.6 Asset-list inconsistencies
- File/section: `README.md` § "Project layout" vs § "Quick start" vs § "Reproducing"
- Issue: The layout description lists `data/_usd_1h.csv  # ADA, BNB, DOGE, LINK, LTC, SHIB, SOL, XRP`. SHIB is in data but never in any default asset list — there is no explanation of why it is there. USDC/FDUSD are mentioned in one place as `stablecoins/` subdir but the actual files live in `data/` directly. The subdirectory is missing.
- Suggestion: state explicitly what the canonical asset set is, list the data files that are NOT used, and remove or create the `stablecoins/` directory.

### 1.7 The `--no-wc` flag is undocumented
- File/section: `README.md` vs `scripts/run_pipeline.py`
- Issue: `scripts/run_pipeline.py` has `--no-wc`, but README does not mention it. "Multiplicative risk-scaler is available but not used in the canonical config" — yet there's no doc on which switch toggles the other canonical config option.
- Suggestion: add a "CLI flags" subsection listing `--assets, --data-dir, --output-dir, --n-boot, --no-wc`.

---

## 2. config/default.yaml

### 2.1 Several fields lack units / one-line semantics
- File/section: `config/default.yaml`
- Issue: `target_grid: [0.20, 0.30, …, 1.00]` — no comment, no mention of whether these are annualized vol fractions or per-state vol budgets. `pipeline.lags: 5` — no units; lag of bars? hours? A new user reading the config cannot tell.
- Suggestion: inline-comment every numeric with `(hours)`, `(annualized fraction)`, etc.

### 2.2 Field `target_drag_pct` is named opaquely
- File/section: `config/default.yaml` § "worstcase"
- Issue: `target_drag_pct: 0.5` with the inline comment "pick smallest eta where |V_wc - V|/|V| >= this". The variable name doesn't match the explanation. New users can't reason about whether to raise or lower it.
- Suggestion: rename to `min_rel_drag_threshold` and explicitly note that lower = pick a less conservative η.

### 2.3 Three undocumented / mismatched things
- File/section: `config/default.yaml`
- Issue: `progress.style: "rich"` — README never says which styles are valid. `hmm.implementation: "log"` — valid options not enumerated. `multiplier_enabled: false` — README mentions the multiplicative risk-scaler but doesn't show how to flip the bit (no CLI flag, requires editing yaml).
- Suggestion: either document these or expose them on the CLI.

### 2.4 `bootstrap` block is not mentioned anywhere in README
- File/section: `config/default.yaml` § "bootstrap"
- Issue: `bootstrap.n_boot` is exposed as `--n-boot`, but `block_size` and `seed` are not surfaced. Beginner has no idea why CIs are wide vs narrow.
- Suggestion: short paragraph in README: "block_size controls effective sample count; larger = wider CIs but less autocorrelation".

---

## 3. docs/README-results.md

### 3.1 Quantitative terms are assumed, not defined
- File/section: `docs/README-results.md` § "Headline" + § "Methodology"
- Issue: "Sharpe", "Max-DD", "P(HMM>B&H)", "block bootstrap", "Δ Sharpe", "DD Reduction", "annualized vol", "vol-zscore", "vol-of-vol" are all used with no glossary. For someone with no quant background, the document is essentially unreadable past the table.
- Suggestion: add a top-of-doc glossary: "Sharpe = mean / std of hourly returns, annualized by ×√8760", "Max-DD = peak-to-trough log-return", "P(HMM>B&H) = fraction of bootstrap samples where HMM Sharpe > B&H Sharpe", "block bootstrap = resample 1-week blocks with replacement to estimate CI".

### 3.2 "Δ Sharpe" and "DD Reduction" lack formulas
- File/section: `docs/README-results.md` § "Headline" table columns
- Issue: Columns `Δ Sharpe` and `DD Reduction` show numbers but never explain how they were computed. From the table I have to guess: Δ = HMM Sharpe − B&H Sharpe; DD reduction = (|B&H DD| − |HMM DD|) / |B&H DD|. This should be stated.
- Suggestion: add a caption under the table with the formula.

### 3.3 USDC/FDUSD placeholder table is misleading
- File/section: `docs/README-results.md` § "Asset classes"
- Issue: "FDUSD skipped due to insufficient data" — but `python scripts/run_pipeline.py ... --assets FDUSD` will throw an obscure loader error rather than a polite "FDUSD has only N hours, need ≥ M".
- Suggestion: validate data length in the loader and error early with a readable message listing the asset and the available bars.

### 3.4 Methodology step 8 is opaque
- File/section: `docs/README-results.md` § "Methodology" step 8
- Issue: "Per-state target vol (joint brute force over K=3 × 9 grid = 729 combos)" — new users don't know what the "9 grid" refers to. Presumably it's `target_grid` from the yaml, but never linked.
- Suggestion: write "9 = `target_grid` size in `config/default.yaml`" and link to it.

---

## 4. docs/lse_api_notes.md

### 4.1 `.env` shows `LSE_KEY`, the doc tells the SDK to read `LSE_API_KEY`
- File/section: `docs/lse_api_notes.md` § "Auth + Rate Limits" + "Python SDK" + ".env"
- Issue: The LSE Python SDK reads `LSE_API_KEY` (per the doc). The committed `.env` exports `LSE_KEY`. The doc explicitly acknowledges this in § "Concrete next steps" but does not actually fix it. So a user following the doc will set `LSE_API_KEY` and find the SDK silently reads nothing.
- Suggestion: either rename to `LSE_API_KEY` in `.env`, or add a `.env_loader` shim that maps one to the other; update README to reflect which.

### 4.2 The doc never says "use LSE when …"
- File/section: `docs/lse_api_notes.md` overall
- Issue: Tons of endpoints, rate limits, plan caps. No upfront "When should I use this vs yfinance vs raw CSV". A beginner has to read the whole doc to find the "useful, but not sufficient on its own" verdict at the bottom.
- Suggestion: TL;DR section at the top: "Use LSE when you need live ticks / long history / 1-second candles; do NOT use it for order book, sentiment, on-chain, or derivatives. For those, see Suggested Fallback Vendors at the bottom."

### 4.3 Verdict is hidden at the bottom
- File/section: `docs/lse_api_notes.md` § "Recommendation for the risk-modul project"
- Issue: The decision-relevant content is the last 20% of the file. Most users will skim the endpoint tables and never reach it.
- Suggestion: move the "Use it for / Do not expect" bullets into a callout box near the top.

### 4.4 Production API key is committed
- File/section: `.env`, `docs/lse_api_notes.md`
- Issue: `.env` is in the repo and contains `lse_live_53ac72f820d0047b2b913a0dfb833ec1`. `docs/lse_api_notes.md` displays redacted samples but the actual production key sits in `.env` and is presumably tracked. There is no `.gitignore` entry visible from the listing — please verify it is ignored, and rotate the key.
- Suggestion: rotate the key, add `echo ".env" >> .gitignore` if missing, ship a `.env.example` with a placeholder.

---

## 5. .env handling

### 5.1 No documentation of which env vars the pipeline actually uses
- File/section: `README.md` (env vars absent) + `.env`
- Issue: `.env` defines `LSE_KEY`. README Quick Start never mentions env vars, never mentions LSE, never mentions `.env`. The pipeline runs to completion without ever reading `.env` (no code in `src/` references `os.environ["LSE_KEY"]` from the live run). So the env var exists, README doesn't mention it, and the pipeline doesn't use it — a triple-mismatch.
- Suggestion: add a "Environment variables" section to README listing each `LSE_API_KEY/ LSE_KEY` and saying "currently informational only; pipeline reads bundled CSV/parquet".

### 5.2 Production credentials in plain text in repo
- File/section: `.env`
- Issue: A live `lse_live_*` key is committed. Whether this is gitignored or not, anyone with shell access to the repo has the key.
- Suggestion: move to a secret manager or vault; keep `.env.example` with `<your-key>` placeholder.

---

## 6. Source code organization

### 6.1 No README pointer to "where do I edit X"
- File/section: `README.md` § "Project layout" vs `src/`
- Issue: `src/` contains three large top-level modules (`multi_asset_runner.py`, `risk_pipeline_worstcase.py`, `risk_pipeline_hourly.py`) plus subpackages `risk_strategy/`, `diagnostics/`, `data_io/`, `utils/`. README only labels `src/multi_asset_runner.py` as the entry point and says `src/risk_pipeline_worstcase.py` is the "canonical BTC pipeline". The relationship between these three top-level files is undocumented. A new dev does not know which one to touch for, e.g., "change transaction costs".
- Suggestion: in `README.md` add a one-liner per source file: `multi_asset_runner.py = CLI per-asset walker`, `risk_pipeline_worstcase.py = BTC reference incl. V_wc`, `risk_pipeline_hourly.py = BTC baseline`, and map "edit transaction cost" → `pipeline.tc_per_side` in config.

### 6.2 scripts/ is underdocumented
- File/section: `scripts/run_pipeline.py`
- Issue: README says CLI lives in `scripts/run_pipeline.py`. But reading that file shows it spawns `src/multi_asset_runner.py` as a subprocess — there are effectively TWO entry points, only one is called out. README's "Pipeline (per asset)" ASCII diagram references internal files (`Build V_wc`) that have no obvious mapping to script names.
- Suggestion: in README add an "Entry points" subsection: `scripts/run_pipeline.py` (CLI wrapper) → `src/multi_asset_runner.py` (per-asset core) → `src/risk_pipeline_worstcase.py` (re-used compute kernel).

### 6.3 Subpackage contents are invisible from README
- File/section: `src/risk_strategy/`, `src/diagnostics/`, `src/data_io/`
- Issue: README lists these as one-line dir entries. New user has to `ls` to know what's inside.
- Suggestion: inline each subpackage README or list the .py files.

---

## 7. Tests

### 7.1 Tests pass, but README's coverage claim is incomplete
- File/section: `README.md` § "Tests" vs `tests/`
- Issue: All 16 tests passed in 0.24s. README says "unit tests for metrics, position-sizing, V_wc" — actually three files exist: `test_metrics.py`, `test_lookahead.py`, `test_worstcase.py`. The "position-sizing" piece lives in `test_lookahead.py` and is not named as such. There is no test for HMM, no integration / pipeline-end-to-end test, no test for the config loader. So the test surface is narrower than the methodology.
- Suggestion: either rename `test_lookahead.py` → `test_position_sizing_lookahead.py` (and split), or update README to say "tests cover: metrics, lookahead bias in position sizing, V_wc monotonicity".

### 7.2 No test asserts the pipeline runs end-to-end
- File/section: `tests/`
- Issue: There is no smoke test that exercises `multi_asset_runner.main` on a one-bar fixture or that round-trips a USDC/FDUSD file.
- Suggestion: add a `test_pipeline_smoke.py` that runs the runner on a 200-row fixture asset and asserts the summary JSON contains the expected keys.

---

## 8. Summary of top 10 "fix first" issues

1. Reconcile the Headline result block with the Expected-output table in README (BTC Sharpe 0.853 vs -0.42 — same asset, two different numbers).
2. Rotate the committed `LSE_KEY` in `.env` and add `.env` to `.gitignore` (and a `.env.example`).
3. Add Quick Start steps for `python -m venv .venv` and `cp .env.example .env`.
4. Document `--no-wc` and any other CLI flags.
5. Add a glossary to `docs/README-results.md` defining Sharpe, Max-DD, P(HMM>B&H), block bootstrap, Δ Sharpe, DD Reduction.
6. Inline-document every numeric in `config/default.yaml` with units.
7. Either rename `LSE_KEY` → `LSE_API_KEY` everywhere or ship a shim; document which env vars the pipeline reads (currently: none from the live code path).
8. Document the source layout: scripts/run_pipeline.py → src/multi_asset_runner.py → src/risk_pipeline_worstcase.py.
9. Add a smoke test that runs the pipeline end-to-end on a tiny fixture.
10. State the "8 of 10 outperforms" claim mechanically (boolean column or sorted rows in the README table).
