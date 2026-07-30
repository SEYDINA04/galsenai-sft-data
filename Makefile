# ════════════════════════════════════════════════════════════════
#  Makefile — interface de la plateforme galsenai-sft-data.
# ════════════════════════════════════════════════════════════════
.DEFAULT_GOAL := help

.PHONY: help setup hooks lint format test converters clean

help:  ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Crée le venv et installe le projet (uv)
	uv venv --python 3.12
	uv pip install -e ".[dev]"

hooks:  ## Installe les hooks git (pre-commit + pre-push)
	uv run pre-commit install --install-hooks
	uv run pre-commit install --hook-type pre-push

lint:  ## Lint (ruff)
	uv run ruff check src tests

format:  ## Formatage (ruff)
	uv run ruff format src tests

test:  ## Tests (pytest)
	uv run pytest

converters:  ## Liste les converters disponibles
	uv run galsenai-sft converters

clean:  ## Nettoie les caches
	rm -rf .pytest_cache .ruff_cache **/__pycache__
