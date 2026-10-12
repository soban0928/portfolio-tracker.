# Full reproducible pipeline. Run from world-cup-predictor/.
PY ?= python
SIMS ?= 100000

.PHONY: install data pipeline quick app test lint format clean

install:            ## install package + dev tools
	$(PY) -m pip install -e ".[dev]"

data:               ## download raw data into data/raw/
	$(PY) scripts/download_data.py

pipeline:           ## train, evaluate and simulate 2018, 2022, 2026 -> artefacts/
	$(PY) scripts/run_pipeline.py --sims $(SIMS)

quick:              ## fast smoke run of the pipeline (few sims)
	$(PY) scripts/run_pipeline.py --sims 2000 --out artefacts_quick

app:                ## launch the dashboard
	streamlit run app/Home.py

test:
	$(PY) -m pytest --cov

lint:
	ruff check . && ruff format --check .

format:
	ruff format . && ruff check --fix .

clean:
	rm -rf artefacts_quick .pytest_cache .ruff_cache .coverage
