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
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

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

.PHONY: backbone-eval
backbone-eval: ## Score holding ground under pushback against the real model (costs API calls; RUNS=5)
	$(PY) scripts/backbone_eval.py --runs $(or $(RUNS),5) --out .backbone-eval.json

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

# --- experimental Monika persona, on its own fresh data dir (DB/lock/socket/config) -------------
# Runs this checkout's code (PYTHONPATH=src) so it also works from a git worktree without its own
# venv: falls back to the main checkout's .venv. Your default ~/.aca agent is never touched.
MONIKA_DIR ?= $(HOME)/.aca-monika
MONIKA_MODEL ?= gpt-6-luna
MAIN_CHECKOUT := $(shell dirname "$$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)")
ACA_BIN := $(if $(wildcard $(CURDIR)/$(BIN)/aca),$(CURDIR)/$(BIN),$(MAIN_CHECKOUT)/$(VENV)/bin)
MONIKA_ACA := ACA_DATA_DIR=$(MONIKA_DIR) PYTHONPATH=$(CURDIR)/src $(ACA_BIN)/aca

.PHONY: monika-init
monika-init: ## Create the Monika data dir + config (never overwrites; links the repo .env)
	@mkdir -p $(MONIKA_DIR)
	@test -f $(MONIKA_DIR)/config.json || { printf '%s\n' \
		'{' \
		'  "identity": { "name": "Monika", "persona": "monika" },' \
		'  "llm": { "provider": "openai", "model": "$(MONIKA_MODEL)", "max_completion_tokens": 512 },' \
		'  "tts": { "provider": "kokoro" }' \
		'}' > $(MONIKA_DIR)/config.json && echo "wrote $(MONIKA_DIR)/config.json"; }
	@if [ ! -e $(MONIKA_DIR)/.env ] && [ -f $(MAIN_CHECKOUT)/.env ]; then \
		ln -s $(MAIN_CHECKOUT)/.env $(MONIKA_DIR)/.env && echo "linked $(MONIKA_DIR)/.env -> $(MAIN_CHECKOUT)/.env"; fi

.PHONY: monika-run
monika-run: monika-init ## Start the Monika agent service (foreground)
	$(MONIKA_ACA) service start

.PHONY: monika-chat
monika-chat: ## Chat with Monika
	$(MONIKA_ACA) chat

.PHONY: monika-voice
monika-voice: ## Chat with Monika by voice (needs the voice extra)
	$(MONIKA_ACA) chat --voice

.PHONY: monika-status
monika-status: ## Show the Monika agent's status
	$(MONIKA_ACA) status

.PHONY: monika-reset
monika-reset: ## Wipe the Monika agent's DB (keeps config); stop its service first
	rm -f $(MONIKA_DIR)/agent.db $(MONIKA_DIR)/agent.db-wal $(MONIKA_DIR)/agent.db-shm

.PHONY: monika-eval
monika-eval: ## Score knowledge boundary, character, and length against the real LLM (costs API calls)
	cd $(MAIN_CHECKOUT) && PYTHONPATH=$(CURDIR)/src $(ACA_BIN)/python $(CURDIR)/scripts/monika_eval.py \
		--data-dir $(MONIKA_DIR)

.PHONY: monika-dialogue
monika-dialogue: ## Print a sample Monika transcript from the real LLM (costs API calls)
	cd $(MAIN_CHECKOUT) && PYTHONPATH=$(CURDIR)/src $(ACA_BIN)/python $(CURDIR)/scripts/monika_eval.py \
		--data-dir $(MONIKA_DIR) --dialogue

.PHONY: clean
clean: ## Remove build artifacts and caches
	rm -rf dist build *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

.PHONY: distclean
distclean: clean ## Also remove the virtualenv
	rm -rf $(VENV)
