# ACA — developer tasks. Run `make help` for the list.
# Targets use a local virtualenv in .venv; PYTHON selects the interpreter used to create it.

PYTHON ?= python3.12
VENV := .venv
BIN := $(VENV)/bin
PY := $(BIN)/python
PIP := $(BIN)/pip

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'

$(VENV): ## Create the virtualenv
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

.PHONY: install
install: $(VENV) ## Create the venv and install the package with dev dependencies
	$(PIP) install -e ".[dev]"

.PHONY: test
test: ## Run the full test suite
	$(BIN)/pytest -q

.PHONY: test-adversarial
test-adversarial: ## Run only the invariant/adversarial tests
	$(BIN)/pytest tests/adversarial -q

.PHONY: lint
lint: ## Lint with Ruff (matches CI)
	$(BIN)/ruff check src tests

.PHONY: format
format: ## Auto-fix lint issues and format with Ruff
	$(BIN)/ruff check --fix src tests
	$(BIN)/ruff format src tests

.PHONY: check
check: lint test ## Run lint and tests (what CI gates on)

.PHONY: build
build: ## Build sdist + wheel and verify with twine
	$(PIP) install --upgrade build twine
	$(PY) -m build
	$(PY) -m twine check dist/*

.PHONY: run
run: ## Start the agent service (foreground)
	$(BIN)/aca service start

.PHONY: chat
chat: ## Open an interactive chat client
	$(BIN)/aca chat

.PHONY: demo
demo: install ## Throwaway agent tuned to message you on its own; drops into chat
	ACA=$(BIN)/aca bash scripts/demo.sh

.PHONY: status
status: ## Show agent status
	$(BIN)/aca status

.PHONY: clean
clean: ## Remove build artifacts and caches
	rm -rf dist build *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

.PHONY: distclean
distclean: clean ## Also remove the virtualenv
	rm -rf $(VENV)
