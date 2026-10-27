"""Prometheus metrics for the server and the client.

Each process serves its metrics at http://<pod>:<KS_METRICS_PORT>/metrics.
Prometheus scrapes that page; Grafana graphs the history.

Every metric object has its own registry, so a test can run a server and
several clients in one process and still read each one's numbers separately.
"""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server
from prometheus_client.process_collector import ProcessCollector

# Per-batch times on CPU run from a few ms to a few s; with injected latency, up to ~1 s more.
BATCH_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
# A round is one local epoch per client: tens of seconds to tens of minutes.
ROUND_BUCKETS = (5, 10, 30, 60, 120, 300, 600, 1200, 3600)


class _Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        ProcessCollector(registry=self.registry)  # process CPU seconds, memory

    def serve(self, port: int, addr: str = "0.0.0.0") -> int:
        """Start the /metrics HTTP endpoint in a background thread. Returns the port."""
        server, _ = start_http_server(port, addr, registry=self.registry)
        return server.server_port


class ServerMetrics(_Metrics):
    def __init__(self):
        super().__init__()
        r = self.registry
        # Progress
        self.round = Gauge("kubesplit_round", "Round currently being trained", registry=r)
        self.training_done = Gauge(
            "kubesplit_training_done", "1 once all rounds have finished", registry=r
        )
        self.clients_joined = Gauge(
            "kubesplit_clients_joined", "Distinct clients that have joined", registry=r
        )
        self.client_joins = Counter(
            "kubesplit_client_joins",
            "Join calls, including rejoins after a client restarts",
            registry=r,
        )
        # Rounds
        self.round_duration = Histogram(
            "kubesplit_round_duration_seconds",
            "Time from a round opening to it closing",
            buckets=ROUND_BUCKETS,
            registry=r,
        )
        self.rounds = Counter(
            "kubesplit_rounds", "Closed rounds, by what closed them", ["reason"], registry=r
        )
        self.round_clients_reporting = Gauge(
            "kubesplit_round_clients_reporting",
            "Clients whose update made it into the last closed round",
            registry=r,
        )
        self.late_updates = Counter(
            "kubesplit_late_updates",
            "Client updates discarded because their round had already closed",
            registry=r,
        )
        # Model quality
        self.test_accuracy = Gauge(
            "kubesplit_test_accuracy_ratio",
            "Test accuracy (0-1) of the global model after the last round",
            registry=r,
        )
        self.train_loss = Gauge(
            "kubesplit_train_loss", "Server-side training loss over the last round", registry=r
        )
        # Traffic
        self.transferred_bytes = Counter(
            "kubesplit_transferred_bytes",
            "Serialized message bytes: activations (up), gradients (down), weights (both ways)",
            ["kind"],
            registry=r,
        )
        # Create every label up front so each series reads 0 instead of "no data".
        for kind in ("activations", "gradients", "weights"):
            self.transferred_bytes.labels(kind)
        for reason in ("quorum", "timeout"):
            self.rounds.labels(reason)
        self.server_samples = Counter(
            "kubesplit_server_samples", "Samples run through the server's top model", registry=r
        )
        # Server load
        self.step_seconds = Histogram(
            "kubesplit_server_step_seconds",
            "Top-model forward+backward time per batch",
            buckets=BATCH_BUCKETS,
            registry=r,
        )
        self.lock_wait_seconds = Histogram(
            "kubesplit_server_lock_wait_seconds",
            "Time a batch queues for the shared top model (server contention)",
            buckets=BATCH_BUCKETS,
            registry=r,
        )


class ClientMetrics(_Metrics):
    def __init__(self):
        super().__init__()
        r = self.registry
        self.round = Gauge("kubesplit_client_round", "Round this client is training", registry=r)
        self.batch_seconds = Histogram(
            "kubesplit_client_batch_seconds",
            "Per-batch time: compute = local forward/backward; "
            "blocked = waiting on the network and the server",
            ["phase"],
            buckets=BATCH_BUCKETS,
            registry=r,
        )
        for phase in ("compute", "blocked"):
            self.batch_seconds.labels(phase)
        self.samples = Counter(
            "kubesplit_client_samples", "Training samples this client has processed", registry=r
        )
        self.round_seconds = Histogram(
            "kubesplit_client_round_seconds",
            "Time to train this client's shard for one round",
            buckets=ROUND_BUCKETS,
            registry=r,
        )
        self.idle_seconds = Histogram(
            "kubesplit_client_idle_seconds",
            "Time spent waiting for the next round to open (the round barrier)",
            buckets=ROUND_BUCKETS,
            registry=r,
        )
