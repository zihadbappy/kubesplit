PY ?= python
COMPOSE ?= docker compose
MODE ?= decoupled
ARGS ?=

.PHONY: install proto test lint fmt baseline baseline-quick local image compose-up compose-both clean

install:
	$(PY) -m pip install -e ".[dev]"

# Regenerate gRPC stubs after editing proto/kubesplit/kubesplit.proto.
proto:
	$(PY) -m grpc_tools.protoc -I proto --python_out=. --pyi_out=. --grpc_python_out=. \
		proto/kubesplit/kubesplit.proto

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

fmt:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

# Real CIFAR-10 (downloads ~170 MB to ./data on first run).
baseline:
	$(PY) -m kubesplit.baseline --mode both --clients 4 --rounds 5 --csv results/baseline.csv $(ARGS)

# Synthetic data; seconds on a laptop CPU. Doubles as a smoke test.
baseline-quick:
	$(PY) -m kubesplit.baseline --mode both --clients 4 --rounds 2 --fake-data $(ARGS)

# Server + 4 clients as separate local processes over gRPC; no Docker needed.
local:
	$(PY) scripts/local_run.py --mode $(MODE) $(ARGS)

image:
	docker build -t kubesplit:dev .

# One mode through Docker Compose: make compose-up MODE=vanilla
compose-up:
	mkdir -p data results
	MODE=$(MODE) $(COMPOSE) up --build --abort-on-container-failure
	$(COMPOSE) down

compose-both:
	$(MAKE) compose-up MODE=vanilla
	$(MAKE) compose-up MODE=decoupled

clean:
	rm -rf .pytest_cache .ruff_cache build *.egg-info
