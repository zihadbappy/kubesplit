PY ?= python
COMPOSE ?= docker compose
MODE ?= decoupled
ARGS ?=
TARGET ?= client-0
KIND_CLUSTER ?= kubesplit
KUBECTX = kind-$(KIND_CLUSTER)
KPS_VERSION ?= 92.14.1

.PHONY: install proto test lint fmt baseline baseline-quick local image compose-up compose-both \
	helm-lint tf-check dashboard kind-up kind-load monitoring prometheus-ui grafana-ui demo \
	pod-kill demo-gif kind-down gke-monitoring clean

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

# The image bakes in CIFAR-10 from data/; download it first if it isn't there.
data/cifar-10-batches-py:
	$(PY) -m kubesplit prepare-data --data-root data

image: data/cifar-10-batches-py
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

# Terraform format + validate for every stack; no credentials or backend needed.
TF_STACKS = $(wildcard infra/*/versions.tf)
tf-check:
	terraform fmt -check -recursive infra
	for f in $(TF_STACKS); do d=$$(dirname $$f); \
		terraform -chdir=$$d init -backend=false -input=false >/dev/null && \
		terraform -chdir=$$d validate || exit 1; done

# Rebuild the chart's Grafana dashboard JSON after editing scripts/make_dashboard.py.
dashboard:
	$(PY) scripts/make_dashboard.py

# Local Kubernetes: make kind-up && make demo
kind-up:
	kind create cluster --config deploy/kind/cluster.yaml --wait 180s

# kind nodes can't see local Docker images; copy ours in.
kind-load: image
	kind load docker-image kubesplit:dev --name $(KIND_CLUSTER)

# Prometheus Operator + Prometheus + Grafana. Run before `demo`: the kind values
# enable our ServiceMonitor, which needs the operator's CRDs.
monitoring:
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts \
		--force-update
	helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
		--version $(KPS_VERSION) -n monitoring --create-namespace \
		-f deploy/kind/monitoring-values.yaml --kube-context $(KUBECTX) --wait --timeout 15m

# Same stack on GKE, into the current kubectl context (after terraform's get_credentials).
gke-monitoring:
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts \
		--force-update
	helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
		--version $(KPS_VERSION) -n monitoring --create-namespace \
		-f deploy/gke/monitoring-values.yaml --wait --timeout 15m

# Prometheus UI at http://localhost:9090 (Ctrl+C to stop).
prometheus-ui:
	kubectl --context $(KUBECTX) -n monitoring port-forward \
		svc/monitoring-kube-prometheus-prometheus 9090:9090

# Grafana at http://localhost:3000, dashboard "KubeSplit" (Ctrl+C to stop).
# Log in as admin; the password is in the monitoring-grafana secret.
grafana-ui:
	kubectl --context $(KUBECTX) -n monitoring port-forward svc/monitoring-grafana 3000:80

# Install (or upgrade) the chart with laptop-sized values; ARGS="--set run.mode=vanilla".
demo: kind-load
	helm upgrade --install kubesplit charts/kubesplit -f deploy/kind/values.yaml \
		--kube-context $(KUBECTX) $(ARGS)
	kubectl --context $(KUBECTX) rollout status deployment/kubesplit-server --timeout=180s
	kubectl --context $(KUBECTX) rollout status statefulset/kubesplit-client --timeout=180s

# Kill a pod mid-round and time the recovery: make pod-kill TARGET=server ARGS="--at-round 5".
pod-kill:
	$(PY) scripts/pod_kill.py $(TARGET) --context $(KUBECTX) $(ARGS)

# Re-record docs/demo.gif and docs/demo.cast on kind (reinstalls the release, ~2 min).
demo-gif: kind-load
	$(PY) scripts/record_demo.py --context $(KUBECTX)

kind-down:
	kind delete cluster --name $(KIND_CLUSTER)

clean:
	rm -rf .pytest_cache .ruff_cache build *.egg-info
