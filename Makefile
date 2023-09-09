PWD      := $(shell pwd)
PYTHON   := poetry run python
PYSEN    := poetry run pysen
export PYTHONPATH=$(PWD)
export OPENAI_API_KEY

.PHONY: get_presentation_summary
get_presentation_summary:
	$(PYTHON) scripts/get_presentation_summary.py

.PHONY: lint
lint:
	$(PYSEN) run lint

.PHONY: format
format:
	$(PYSEN) run format
