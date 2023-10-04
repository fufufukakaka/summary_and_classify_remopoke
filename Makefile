PWD        := $(shell pwd)
PYTHON     := poetry run python
PYSEN      := poetry run pysen
PYTHONPATH := $(PWD)
export VIDEO_URL
export OPENAI_API_KEY


.PHONY: get_presentation_summary
get_presentation_summary_for_a_video:
	$(PYTHON) scripts/get_presentation_summary_langchain.py --video_url $(VIDEO_URL)

get_presentation_summary:
	$(PYTHON) scripts/get_presentation_summary_langchain.py

.PHONY: lint
lint:
	$(PYSEN) run lint

.PHONY: format
format:
	$(PYSEN) run format
