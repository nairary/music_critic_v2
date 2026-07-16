.PHONY: help test compile check legacy-check

PYTHON ?= python

help:
	@echo "Available targets:"
	@echo "  test         Run the test suite"
	@echo "  compile      Compile Python sources"
	@echo "  check        Run tests and compilation"
	@echo "  legacy-check Verify that the legacy repository matches the snapshot"

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

compile:
	$(PYTHON) -m compileall src

check: test compile

legacy-check:
	$(PYTHON) scripts/check_legacy_unchanged.py
