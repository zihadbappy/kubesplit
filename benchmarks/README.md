# Benchmarks

Measured on GKE on 2026-10-30. The numbers are in [gke/results.md](gke/results.md),
the raw data next to it, and the charts in the main [README](../README.md#results).

## Setup

| | |
|---|---|
| Cluster | GKE 1.36.4, zonal (us-central1-a), 3 × `n2-standard-4` Spot VMs, from [infra/cluster](../infra/cluster) |
| Image | `b2702f2` (training runs, first pod-kill set), `44f0a32` (faster-rejoin pod kills): CPU-only torch 2.11, CIFAR-10 baked in |
| Pods | server 2 CPU / 2 GiB, 4 clients 1 CPU / 1.5 GiB each, requests = limits (Guaranteed QoS) |
| Data | CIFAR-10, IID shards of 12,500 training images per client, test on all 10,000 |
| Training | batch 64, SGD lr 0.05 with momentum 0.9, 5 rounds of one local epoch, FedAvg after each |
| Latency | `run.netDelayMs` = 0, 25, 50, 100: tc netem delays every packet a client sends, so that much more round-trip time to the server |
| Monitoring | kube-prometheus-stack on the same nodes, scraping every 5 s |

## Method

[`scripts/benchmark.py`](../scripts/benchmark.py) runs each configuration from a
fresh `helm install` (an upgrade lets old client pods join the new run before
they are replaced, inflating its first rounds), waits for the server to report
training done, and records the server's per-round CSV plus a summary row.
Configurations are interleaved (repeat 1 of all, then repeat 2, ...) so drift in
the cloud doesn't land on one of them. Each ran 3 times; all 24 runs were clean:
no pod restarted and every client joined once.

- **seconds per round**: mean of rounds 2–5. Round 1 also covers pods starting
  and loading data.
- **compute / blocked per batch**: client-side histograms. Compute is the
  client's own forward and backward; blocked is time waiting on the network
  and the server.
- Ranges are the minimum and maximum over the 3 runs.

[`scripts/pod_kill.py`](../scripts/pod_kill.py) deletes a pod 5 s into a round
on a 40-round decoupled run and times each stage of recovery.

## Reproduce

```bash
cd infra/cluster && terraform apply && $(terraform output -raw get_credentials) && cd ../..
make gke-monitoring
python scripts/benchmark.py --modes vanilla,decoupled --delays 0,25,50,100 --repeats 3 \
  --rounds 5 --out benchmarks/gke -- -f deploy/gke/values.yaml \
  --set-string image.repository=$IMAGE_REPOSITORY --set-string image.tag=<sha>
python scripts/plot_results.py benchmarks/gke --img docs/img
cd infra/cluster && terraform destroy
```

The 24 runs, 6 pod kills and 3 more after the rejoin change took about
2.6 hours of cluster time.
