CONFIG ?= configs/react_baseline.local.yaml
TASK   ?= task_1
LIMIT  ?= 5

.PHONY: install tui run run-bench status inspect lint fmt test

install:
	uv sync

tui:
	uv run dabench tui --config $(CONFIG)

run:
	uv run dabench run-task $(TASK) --config $(CONFIG)

run-bench:
	uv run dabench run-benchmark --config $(CONFIG) --limit $(LIMIT)

status:
	uv run dabench status --config $(CONFIG)

inspect:
	uv run dabench inspect-task $(TASK) --config $(CONFIG)

lint:
	uv run --with ruff ruff check src

fmt:
	uv run --with ruff ruff format src

test:
	uv run pytest
