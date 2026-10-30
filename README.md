# KubeSplit

Kubernetes-native split learning. A gRPC server holds the top of a CNN;
N client pods hold the bottom and train on their own CIFAR-10 shard. Two modes:

- **vanilla**: the client sends activations, then waits for gradients on every batch.
- **decoupled**: the client trains its bottom with a local auxiliary head
  (Decoupled Greedy Learning) and streams activations one way.

After each round the client bottoms are averaged with FedAvg, weighted by sample count.
A round closes when a quorum of clients has reported or a timeout fires.

> Status: Days 1–2 done (single-process baseline; gRPC services; Docker Compose).
> Helm, Terraform and CI are coming next.

## Quick start

```bash
pip install -e ".[dev]"
make test             # pytest, ~1 min on CPU, no download
make baseline-quick   # both modes in one process, synthetic data
make baseline         # both modes in one process, real CIFAR-10 -> results/baseline.csv
make local MODE=vanilla    # server + 4 client processes over gRPC, no Docker
make compose-both          # server + 4 client containers, vanilla then decoupled
```

Without `make` (e.g. Windows): `python -m kubesplit {server,client,baseline} --help`
and `python scripts/local_run.py --help`.

## Run on Kubernetes (kind)

Needs Docker, [kind](https://kind.sigs.k8s.io/), kubectl and Helm. Uses synthetic data,
so nothing is downloaded; fits a 6-CPU / 7 GiB Docker Desktop.

```bash
make kind-up        # one-node cluster in Docker
make monitoring     # kube-prometheus-stack: Prometheus Operator, Prometheus, Grafana
make demo           # build + load the image, helm install, wait until all pods are ready
kubectl logs -f deployment/kubesplit-server   # watch rounds close
make prometheus-ui  # http://localhost:9090, try: kubesplit_round or up{job=~"kubesplit-.*"}
make grafana-ui     # http://localhost:3000, dashboard "KubeSplit"
make kind-down      # delete the cluster
```

The chart's `ServiceMonitor` (enabled in `deploy/kind/values.yaml`) makes Prometheus
scrape the server and every client pod every 5 s, labelled `component="server"|"client"`.
Its `grafanaDashboard` ConfigMap is loaded by Grafana's sidecar as the **KubeSplit**
dashboard: run progress, client compute vs blocked time per batch, traffic by message
kind, server queueing, and each pod's CPU, throttling and memory against its limits.
Log in as `admin`; the password is in a secret:

```bash
kubectl -n monitoring get secret monitoring-grafana -o jsonpath="{.data.admin-password}" | base64 -d
```

```powershell
[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String((kubectl -n monitoring get secret monitoring-grafana -o jsonpath="{.data.admin-password}")))
```

Without `make`, run the commands under those targets in the `Makefile`. Server and
clients stay up after training (`KS_EXIT_ON_DONE=false`) so their metrics remain
scrapeable. A new run starts whenever a `run.*` value changes:
`helm upgrade kubesplit charts/kubesplit -f deploy/kind/values.yaml --set run.mode=vanilla`.

## How a round works

```
client                                     server
  Join(shard)  ───────────────────────────▶  returns RunConfig (mode, partition, lr, ...)
  WaitRound(last) ────────────────────────▶  long-poll until a newer round opens
               ◀────────────────────────────  round r + global bottom weights
  vanilla:   Step(acts)  ─────────────────▶  top fwd/bwd
             ◀──────────────── grad at cut   (one round trip per batch)
  decoupled: StreamActivations(acts...) ──▶  top fwd/bwd, nothing sent back
  SubmitUpdate(weights, n) ───────────────▶  quorum reached or timeout -> FedAvg,
                                             evaluate, write results row, open r+1
```

The server owns the config, so clients only need `KS_SERVER` and a shard. The
shard defaults to the hostname's trailing ordinal (`kubesplit-client-3` → 3).
Late updates for a closed round are discarded; the straggler joins the next round.

**When pods die.** A replacement client pod rejoins and trains the current round.
The server keeps no state on disk, so a replacement server starts a new run from
round 1. Each server process has a random `server_id`, which Join and WaitRound
return. When a client sees a different ID, or the server drops a call
(`UNAVAILABLE`), it joins again in-process instead of crashing, so Kubernetes never
restarts it. Both roles exit on SIGTERM within about a second.
Tensors travel as dtype + shape + raw bytes (`kubesplit/tensors.py`), never pickle.

## Configuration

Every flag also reads a `KS_*` environment variable: `KS_MODE`, `KS_CLIENTS`,
`KS_ROUNDS`, `KS_QUORUM_PCT`, `KS_ROUND_TIMEOUT`, `KS_PARTITION`, `KS_ALPHA`,
`KS_DATASET` (`cifar10`/`fake`), `KS_MAX_BATCHES`, `KS_SERVER`, `KS_SHARD`, `KS_THREADS`,
`KS_METRICS_PORT` (default 8000; 0 turns metrics off).
Torch sizes its thread pool from the container's cgroup CPU quota, not the host core count.

Results: one CSV per run in `results/`, one row per round: test accuracy, train
loss and accuracy, `bytes_up` / `bytes_down` / `bytes_sync` (serialized message sizes),
round time, and the number of clients that reported.

## Metrics

Every process serves Prometheus metrics at `:8000/metrics` (`kubesplit/metrics.py`).

| Server metric | Type | Answers |
|---|---|---|
| `kubesplit_round`, `kubesplit_training_done` | gauge | Where is training? |
| `kubesplit_round_duration_seconds` | histogram | How long does a round take? |
| `kubesplit_rounds_total{reason}` | counter | Do rounds close by quorum or by timeout? |
| `kubesplit_round_clients_reporting` | gauge | How many clients made the last round? |
| `kubesplit_late_updates_total` | counter | How often is a straggler's work thrown away? |
| `kubesplit_client_joins_total`, `kubesplit_clients_joined` | counter, gauge | Did a killed client come back? |
| `kubesplit_test_accuracy_ratio`, `kubesplit_train_loss` | gauge | Is the model learning? |
| `kubesplit_transferred_bytes_total{kind}` | counter | Bytes of activations, gradients, weights |
| `kubesplit_server_samples_total` | counter | Server throughput (use `rate()`) |
| `kubesplit_server_step_seconds` | histogram | Top-model compute per batch |
| `kubesplit_server_lock_wait_seconds` | histogram | How long batches queue for the shared top model |

| Client metric | Type | Answers |
|---|---|---|
| `kubesplit_client_batch_seconds{phase}` | histogram | Per batch: `compute` (local) vs `blocked` (network + server) |
| `kubesplit_client_samples_total` | counter | Client throughput (use `rate()`) |
| `kubesplit_client_round_seconds` | histogram | Time to train one round's shard |
| `kubesplit_client_idle_seconds` | histogram | Time waiting at the round barrier |
| `kubesplit_client_round` | gauge | Round this client is on |

## Layout

| Path | What |
|---|---|
| `proto/kubesplit/kubesplit.proto` | Wire protocol (`make proto` regenerates `kubesplit/*_pb2*.py`) |
| `kubesplit/model.py` | `ClientBottom` (3x32x32 → 64x8x8), `AuxHead`, `ServerTop` |
| `kubesplit/data.py` | CIFAR-10 loading, deterministic IID and Dirichlet(α) sharding |
| `kubesplit/split.py` | `ClientWorker` / `ServerWorker` (independent of transport), `fedavg` |
| `kubesplit/rounds.py` | `RoundCoordinator`: quorum, timeout, FedAvg, late-update rejection |
| `kubesplit/server.py`, `client.py` | gRPC services |
| `kubesplit/tensors.py` | Tensor ↔ protobuf without pickle |
| `kubesplit/baseline.py` | Single-process reference run of both modes |
| `Dockerfile`, `docker-compose.yml` | One CPU-only image for every role; 1 server + 4 clients |
| `charts/kubesplit/` | Helm chart; settings in `values.yaml`, validated by `values.schema.json` |
| `scripts/make_dashboard.py` | Builds the chart's Grafana dashboard (`charts/kubesplit/dashboards/`) |
| `deploy/kind/` | Local cluster config and laptop-sized chart values |

Each sample crosses the cut as 64×8×8 float32 (16 KiB). Vanilla also returns
a gradient of the same size, so it moves about 2× the bytes of decoupled mode.
The interesting number is wall-clock time under network latency.
