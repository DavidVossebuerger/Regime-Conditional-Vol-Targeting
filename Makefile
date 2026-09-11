.PHONY: help install install-dev test test-fast clean run-crypto run-equities run-russell lint format

# --- help ---
help:
	@echo "Risk-Management pipeline — common commands:"
	@echo "  make install      — install runtime deps"
	@echo "  make install-dev  — install runtime + dev deps (pytest)"
	@echo "  make test         — run unit tests"
	@echo "  make lint         — basic syntax check"
	@echo "  make run-crypto       — full crypto pipeline (BTC, ETH, ...)"
	@echo "  make run-equities     — equities pipeline (default 9-ticker basket, needs LSE_API_KEY)"
	@echo "  make run-russell      — Russell 2000 top-200 (basket from local CSV, no LSE needed)"
	@echo "  make clean        — remove generated outputs + caches"

# --- setup ---
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest

# --- quality ---
test:
	python -m pytest tests/ -v

lint:
	python -m py_compile src/multi_asset_runner.py
	python -m py_compile src/equities_runner.py
	python -m py_compile src/data_io/lse_loader.py

# --- pipelines ---
run-crypto:
	python src/multi_asset_runner.py --assets BTC,ETH,ADA,BNB,DOGE,LINK,LTC,SOL,XRP \
	    --output-dir outputs/multi_asset

run-equities:
	python src/equities_runner.py

run-russell:
	python src/equities_runner.py --from-csv data/russell2000_top200.csv \
	    --output-dir outputs/russell2000_top200 --n-boot 500 \
	    --tc-per-side 0.0010

# --- cleanup ---
clean:
	rm -rf outputs/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

# --- analysis ---
stats:
	python3 tests/statistical_rigor.py

report:
	python3 tests/build_html_report.py
