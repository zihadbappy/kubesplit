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
Tensors travel as dtype + shape + raw bytes (`kubesplit/tensors.py`), never pickle.

## Configuration

Every flag also reads a `KS_*` environment variable: `KS_MODE`, `KS_CLIENTS`,
`KS_ROUNDS`, `KS_QUORUM_PCT`, `KS_ROUND_TIMEOUT`, `KS_PARTITION`, `KS_ALPHA`,
`KS_DATASET` (`cifar10`/`fake`), `KS_MAX_BATCHES`, `KS_SERVER`, `KS_SHARD`, `KS_THREADS`.
Torch sizes its thread pool from the container's cgroup CPU quota, not the host core count.

Results: one CSV per run in `results/`, one row per round: test accuracy, train
loss and accuracy, `bytes_up` / `bytes_down` / `bytes_sync` (serialized message sizes),
round time, and the number of clients that reported.

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

Each sample crosses the cut as 64×8×8 float32 (16 KiB). Vanilla also returns
a gradient of the same size, so it moves about 2× the bytes of decoupled mode.
The interesting number is wall-clock time under network latency.
