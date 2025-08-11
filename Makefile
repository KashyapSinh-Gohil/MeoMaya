PY?=python
VENV=.venv

.PHONY: venv install install-dev run run-api test fmt lint docs

venv:
	$(PY) -m venv $(VENV)
	. $(VENV)/bin/activate; pip install --upgrade pip setuptools wheel

install: venv
	. $(VENV)/bin/activate; pip install -r meomaya/requirements.txt

install-dev: venv
	. $(VENV)/bin/activate; pip install -r requirements-dev.txt -r meomaya/requirements.txt

run-api:
	. $(VENV)/bin/activate; uvicorn meomaya.api.server:app --host 0.0.0.0 --port 8000

run-cli:
	. $(VENV)/bin/activate; PYTHONPATH=. python -m meomaya.cli.meomaya_cmd "Hello from MeoMaya!" --mode text

test:
	. $(VENV)/bin/activate; PYTHONPATH=. pytest -q meomaya/tests/test_all.py

fmt:
	. $(VENV)/bin/activate; black meomaya && isort meomaya

lint:
	. $(VENV)/bin/activate; flake8 meomaya


