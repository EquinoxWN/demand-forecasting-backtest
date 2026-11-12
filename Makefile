.PHONY: setup lint test download run features bench audit ci

PY ?= python

setup:
	$(PY) -m pip install --upgrade pip && $(PY) -m pip install -e ".[dev]"

# ruff, ruff format and mypy --strict.
lint:
	$(PY) -m ruff check . && $(PY) -m ruff format --check . && $(PY) -m mypy

# Tests on small synthetic M5-format data (no download needed).
test:
	$(PY) -m pytest -q

# Download the M5 archive (about 50 MB, SHA-256 checked) into data/.
download:
	dfb download

# Seasonal-naive baseline and challengers on the last 28 days of store CA_1.
run: download
	dfb backtest --data data/m5.zip --stores CA_1

# Item-level feature table of store CA_1.
features: download
	dfb features --data data/m5.zip --stores CA_1 --level item --out data/features_item_CA_1.csv.gz

bench:
	@echo "M2/M3: rolling-origin backtests, ETS/ARIMA/LightGBM and interval coverage are not built yet"

# Known vulnerabilities in the installed Python dependencies.
audit:
	$(PY) -m pip_audit --skip-editable --cache-dir .tmp/pip-audit

ci: setup lint test
