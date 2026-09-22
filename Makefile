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

# --- experimental tsundere persona, on its own fresh data dir (DB/lock/socket/config) -----------
# Runs this checkout's code (PYTHONPATH=src) so it also works from a git worktree without its own
# venv: falls back to the main checkout's .venv. Your default ~/.aca agent is never touched.
TSUNDERE_DIR ?= $(HOME)/.aca-tsundere
TSUNDERE_MODEL ?= gpt-5.6-luna
MAIN_CHECKOUT := $(shell dirname "$$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null)")
ACA_BIN := $(if $(wildcard $(CURDIR)/$(BIN)/aca),$(CURDIR)/$(BIN),$(MAIN_CHECKOUT)/$(VENV)/bin)
TSUNDERE_ACA := ACA_DATA_DIR=$(TSUNDERE_DIR) PYTHONPATH=$(CURDIR)/src $(ACA_BIN)/aca

.PHONY: tsundere-init
tsundere-init: ## Create the tsundere data dir + config (never overwrites; links the repo .env)
	@mkdir -p $(TSUNDERE_DIR)
	@test -f $(TSUNDERE_DIR)/config.json || { printf '%s\n' \
		'{' \
		'  "identity": { "persona": "tsundere" },' \
		'  "llm": { "provider": "openai", "model": "$(TSUNDERE_MODEL)", "max_tokens": 512 },' \
		'  "tts": { "provider": "kokoro" }' \
		'}' > $(TSUNDERE_DIR)/config.json && echo "wrote $(TSUNDERE_DIR)/config.json"; }
	@if [ ! -e $(TSUNDERE_DIR)/.env ] && [ -f $(MAIN_CHECKOUT)/.env ]; then \
		ln -s $(MAIN_CHECKOUT)/.env $(TSUNDERE_DIR)/.env && echo "linked $(TSUNDERE_DIR)/.env -> $(MAIN_CHECKOUT)/.env"; fi

.PHONY: tsundere-run
tsundere-run: tsundere-init ## Start the tsundere agent service (foreground)
	$(TSUNDERE_ACA) service start

.PHONY: tsundere-chat
tsundere-chat: ## Chat with the tsundere agent
	$(TSUNDERE_ACA) chat

.PHONY: tsundere-voice
tsundere-voice: ## Chat with the tsundere agent by voice (needs the voice extra)
	$(TSUNDERE_ACA) chat --voice

.PHONY: tsundere-status
tsundere-status: ## Show the tsundere agent's status
	$(TSUNDERE_ACA) status

.PHONY: tsundere-reset
tsundere-reset: ## Wipe the tsundere agent's DB (keeps config); stop its service first
	rm -f $(TSUNDERE_DIR)/agent.db $(TSUNDERE_DIR)/agent.db-wal $(TSUNDERE_DIR)/agent.db-shm

.PHONY: tsundere-eval
tsundere-eval: ## Run the persona eval corpus against the configured real LLM
	cd $(MAIN_CHECKOUT) && PYTHONPATH=$(CURDIR)/src $(ACA_BIN)/python $(CURDIR)/scripts/persona_eval.py \
		--data-dir $(TSUNDERE_DIR)

.PHONY: tsundere-dialogue
tsundere-dialogue: ## Print a sample casual-to-serious transcript from the configured real LLM
	cd $(MAIN_CHECKOUT) && PYTHONPATH=$(CURDIR)/src $(ACA_BIN)/python $(CURDIR)/scripts/persona_eval.py \
		--data-dir $(TSUNDERE_DIR) --dialogue

.PHONY: clean
clean: ## Remove build artifacts and caches
	rm -rf dist build *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

.PHONY: distclean
distclean: clean ## Also remove the virtualenv
	rm -rf $(VENV)
