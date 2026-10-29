PY ?= python
COMPOSE ?= docker compose
MODE ?= decoupled
ARGS ?=
KIND_CLUSTER ?= kubesplit
KUBECTX = kind-$(KIND_CLUSTER)

.PHONY: install proto test lint fmt baseline baseline-quick local image compose-up compose-both \
	helm-lint kind-up kind-load demo kind-down clean

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

helm-lint:
	helm lint charts/kubesplit --strict
	helm lint charts/kubesplit --strict -f deploy/kind/values.yaml

# Local Kubernetes: make kind-up && make demo
kind-up:
	kind create cluster --config deploy/kind/cluster.yaml --wait 180s

# kind nodes can't see local Docker images; copy ours in.
kind-load: image
	kind load docker-image kubesplit:dev --name $(KIND_CLUSTER)

# Install (or upgrade) the chart with laptop-sized values; ARGS="--set run.mode=vanilla".
demo: kind-load
	helm upgrade --install kubesplit charts/kubesplit -f deploy/kind/values.yaml \
		--kube-context $(KUBECTX) $(ARGS)
	kubectl --context $(KUBECTX) rollout status deployment/kubesplit-server --timeout=180s
	kubectl --context $(KUBECTX) rollout status statefulset/kubesplit-client --timeout=180s

kind-down:
	kind delete cluster --name $(KIND_CLUSTER)

clean:
	rm -rf .pytest_cache .ruff_cache build *.egg-info
