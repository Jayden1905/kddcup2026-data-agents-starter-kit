CONFIG ?= configs/react_baseline.local.yaml
TASK   ?= task_11
LIMIT  ?= 50
TEAM   ?= 1347
VERSION ?= v1
IMAGE  ?= $(TEAM):$(VERSION)

.PHONY: install tui run run-bench status inspect lint fmt test docker-build docker-test docker-save

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

docker-build:
	podman build --no-cache --platform linux/amd64 -t $(IMAGE) .

docker-test:
	mkdir -p ./local_output
	podman run --rm \
		-e MODEL_API_URL="$$MODEL_API_URL" \
		-e MODEL_API_KEY="$$MODEL_API_KEY" \
		-e MODEL_NAME="$$MODEL_NAME" \
		-v "$$(pwd)/data/public/input:/input:ro" \
		-v "$$(pwd)/local_output:/output" \
		$(IMAGE)

docker-save:
	podman save $(IMAGE) | gzip > $(TEAM)_$(VERSION).tar.gz
