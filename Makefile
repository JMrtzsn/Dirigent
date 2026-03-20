.PHONY: install dev lint test clean run help

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

venv: ## Create virtual environment
	python3 -m venv $(VENV)

install: venv ## Install production dependencies
	$(PIP) install -e .

dev: venv ## Install all dependencies (dev included)
	$(PIP) install -e ".[dev]"

lint: ## Run ruff linter
	$(RUFF) check src/ tests/

lint-fix: ## Run ruff with auto-fix
	$(RUFF) check --fix src/ tests/

format: ## Format code with ruff
	$(RUFF) format src/ tests/

test: ## Run test suite
	$(PYTEST) -v

test-cov: ## Run tests with coverage
	$(PYTEST) -v --tb=short

check: lint test ## Run lint + tests (CI gate)

run: ## Run Dirigent with a sample objective
	$(PYTHON) -m dirigent.cli "Refactor the auth module" --verbose

clean: ## Remove build artifacts and venv
	rm -rf $(VENV) dist/ build/ *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
