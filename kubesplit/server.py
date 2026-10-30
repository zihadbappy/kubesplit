"""gRPC server: owns the top model, runs FedAvg rounds, writes results.

    python -m kubesplit server --mode decoupled --clients 4 --rounds 5

Every option also reads from a KS_* environment variable (see --help).
"""

import argparse
import csv
import logging
import os
import threading
import time
import uuid
from concurrent import futures
from dataclasses import dataclass

import grpc
import torch
from torch.utils.data import DataLoader, Dataset

from kubesplit import kubesplit_pb2 as pb
from kubesplit import kubesplit_pb2_grpc as pb_grpc
from kubesplit.data import load_dataset
from kubesplit.metrics import ServerMetrics
from kubesplit.rounds import RoundClosed, RoundCoordinator, State
from kubesplit.runtime import (
    GRPC_OPTIONS,
    LONG_POLL_SECONDS,
    add_common_args,
    device,
    env,
    idle_forever,
    parse_bool,
    setup,
)
from kubesplit.split import ServerWorker, client_modules, evaluate
from kubesplit.tensors import from_proto, state_from_proto, state_to_proto, to_proto

log = logging.getLogger("kubesplit.server")


@dataclass
class RoundTraffic:
    """Per-round counters. Bytes are serialized protobuf message sizes."""

    bytes_up: int = 0  # activations + labels, client -> server
    bytes_down: int = 0  # gradients, server -> client (vanilla only)
    bytes_sync: int = 0  # model weights, both directions
    samples: int = 0
    loss_sum: float = 0.0
    correct: int = 0


class SplitLearningServicer(pb_grpc.SplitLearningServicer):
    def __init__(
        self,
        cfg: pb.RunConfig,
        rounds: int,
        test_set: Dataset,
        quorum_pct: float = 100.0,
        round_timeout: float = 600.0,
        results_path: str | None = None,
        poll_seconds: float = LONG_POLL_SECONDS,
        device: str = "cpu",
    ):
        self.cfg = cfg
        self.server_id = uuid.uuid4().hex[:12]  # new on every start: a restart is a new run
        self.poll_seconds = poll_seconds
        self.device = device
        self.results_path = results_path
        self.rows: list[dict] = []

        torch.manual_seed(cfg.seed)
        self.global_model = client_modules(cfg.mode).to(device)
        self.worker = ServerWorker(cfg.lr, device)
        self.test_loader = DataLoader(test_set, batch_size=256)
        self._keys = set(self.global_model.state_dict())

        self._top_lock = threading.Lock()  # one batch at a time through the top model
        self._lock = threading.Lock()  # everything below
        self._traffic = RoundTraffic()
        self._state_cache: tuple[int, pb.StateDict] | None = None
        self.clients: dict[str, int] = {}  # client_id -> shard
        self._told_done: set[str] = set()
        self._started = time.monotonic()
        self._done_at: float | None = None

        initial = {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()}
        self.coord = RoundCoordinator(
            initial,
            expected_clients=cfg.num_clients,
            total_rounds=rounds,
            quorum_pct=quorum_pct,
            round_timeout=round_timeout,
            on_close=self._on_round_closed,
        )

        self.metrics = m = ServerMetrics()
        # Read at scrape time, so they are never stale.
        m.round.set_function(lambda: self.coord.round)
        m.training_done.set_function(lambda: float(self.coord.done))
        m.clients_joined.set_function(lambda: len(self.clients))

    # -- RPCs --------------------------------------------------------------
    def Join(self, request, context):
        if not 0 <= request.shard < self.cfg.num_clients:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"shard {request.shard} outside [0, {self.cfg.num_clients})",
            )
        with self._lock:
            clash = [c for c, s in self.clients.items() if s == request.shard]
            self.clients[request.client_id] = request.shard
        self.metrics.client_joins.inc()
        if clash and clash != [request.client_id]:
            log.warning("shard %d claimed by %s and %s", request.shard, clash, request.client_id)
        log.info("client %s joined with shard %d", request.client_id, request.shard)
        return pb.JoinReply(config=self.cfg, server_id=self.server_id)

    def WaitRound(self, request, context):
        if request.client_id not in self.clients:
            # Joined a previous server, not this one: answer at once, so the
            # client sees our server_id and joins again instead of waiting
            # for a round number from the old run.
            return pb.RoundInfo(round=0, server_id=self.server_id)
        rnd, done, state = self.coord.wait(request.last_round, self.poll_seconds)
        if done:
            with self._lock:
                self._told_done.add(request.client_id)
            return pb.RoundInfo(round=rnd, done=True, server_id=self.server_id)
        reply = pb.RoundInfo(round=rnd, server_id=self.server_id)
        if state is not None:
            reply.global_state.CopyFrom(self._serialized_state(rnd, state))
            self._count_sync(reply.ByteSize())
        return reply

    def Step(self, request, context):
        if self.cfg.mode != "vanilla":
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, "decoupled mode: use StreamActivations"
            )
        h, y = from_proto(request.activations), from_proto(request.labels)
        res = self._train_top(h, y, want_grad=True)
        reply = pb.Gradients(grad=to_proto(res.grad), loss=res.loss, correct=res.correct)
        self._count(request.ByteSize(), reply.ByteSize(), y.numel(), res.loss, res.correct)
        return reply

    def StreamActivations(self, request_iterator, context):
        if self.cfg.mode != "decoupled":
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "vanilla mode: use Step")
        summary = pb.StreamSummary()
        for request in request_iterator:
            h, y = from_proto(request.activations), from_proto(request.labels)
            res = self._train_top(h, y, want_grad=False)
            n = y.numel()
            self._count(request.ByteSize(), 0, n, res.loss, res.correct)
            summary.batches += 1
            summary.samples += n
            summary.loss_sum += res.loss * n
            summary.correct += res.correct
        return summary

    def SubmitUpdate(self, request, context):
        state = state_from_proto(request.state)
        if set(state) != self._keys:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "state dict keys do not match model")
        self._count_sync(request.ByteSize())
        accepted = self.coord.submit(request.client_id, request.round, state, request.num_samples)
        if not accepted:
            self.metrics.late_updates.inc()
            log.info("discarded late update from %s for round %d", request.client_id, request.round)
        return pb.UpdateAck(accepted=accepted, current_round=self.coord.round)

    # -- internals ---------------------------------------------------------
    def _train_top(self, h: torch.Tensor, y: torch.Tensor, want_grad: bool):
        t0 = time.perf_counter()
        with self._top_lock:
            t1 = time.perf_counter()
            res = self.worker.step(h, y, want_grad=want_grad)
        self.metrics.lock_wait_seconds.observe(t1 - t0)
        self.metrics.step_seconds.observe(time.perf_counter() - t1)
        return res

    def _count(self, up: int, down: int, n: int, loss: float, correct: int) -> None:
        with self._lock:
            t = self._traffic
            t.bytes_up += up
            t.bytes_down += down
            t.samples += n
            t.loss_sum += loss * n
            t.correct += correct
        m = self.metrics
        m.transferred_bytes.labels("activations").inc(up)
        m.transferred_bytes.labels("gradients").inc(down)
        m.server_samples.inc(n)

    def _count_sync(self, nbytes: int) -> None:
        with self._lock:
            self._traffic.bytes_sync += nbytes
        self.metrics.transferred_bytes.labels("weights").inc(nbytes)

    def _serialized_state(self, rnd: int, state: State) -> pb.StateDict:
        """Every client fetches the same weights each round; serialize once."""
        with self._lock:
            if self._state_cache is None or self._state_cache[0] != rnd:
                self._state_cache = (rnd, state_to_proto(state))
            return self._state_cache[1]

    def _on_round_closed(self, info: RoundClosed, state: State) -> None:
        self.global_model.load_state_dict(state)
        with self._top_lock:
            acc = evaluate(
                self.global_model["bottom"], self.worker.top, self.test_loader, self.device
            )
        with self._lock:
            t, self._traffic = self._traffic, RoundTraffic()
        row = {
            "mode": self.cfg.mode,
            "clients": self.cfg.num_clients,
            "round": info.round,
            "reporting": len(info.clients),
            "close_reason": info.reason,
            "test_acc": round(acc, 2),
            "train_loss": round(t.loss_sum / max(t.samples, 1), 4),
            "train_acc": round(100.0 * t.correct / max(t.samples, 1), 2),
            "bytes_up": t.bytes_up,
            "bytes_down": t.bytes_down,
            "bytes_sync": t.bytes_sync,
            "round_seconds": round(info.seconds, 3),
            "elapsed_seconds": round(time.monotonic() - self._started, 3),
        }
        self.rows.append(row)
        self._append_csv(row)
        m = self.metrics
        m.round_duration.observe(info.seconds)
        m.rounds.labels(info.reason).inc()
        m.round_clients_reporting.set(len(info.clients))
        m.test_accuracy.set(acc / 100.0)
        m.train_loss.set(row["train_loss"])
        log.info(
            "round %d closed by %s (%d/%d clients) | test %.1f%% | "
            "up %.1f MiB down %.1f MiB | %.1fs",
            info.round,
            info.reason,
            len(info.clients),
            self.cfg.num_clients,
            acc,
            t.bytes_up / 2**20,
            t.bytes_down / 2**20,
            info.seconds,
        )

    def _append_csv(self, row: dict) -> None:
        if not self.results_path:
            return
        new = not os.path.exists(self.results_path)
        with open(self.results_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)

    def finished(self, grace: float) -> bool:
        """Training is over and every client has heard, or `grace` s have passed."""
        if not self.coord.done:
            return False
        with self._lock:
            if self._done_at is None:
                self._done_at = time.monotonic()
            all_told = set(self.clients) <= self._told_done
            return all_told or time.monotonic() - self._done_at > grace


class Server:
    def __init__(self, servicer: SplitLearningServicer, port: int = 50051):
        self.servicer = servicer
        # Each client can hold one long-poll or stream open, plus a unary call.
        workers = 4 * servicer.cfg.num_clients + 8
        self.grpc = grpc.server(futures.ThreadPoolExecutor(workers), options=GRPC_OPTIONS)
        pb_grpc.add_SplitLearningServicer_to_server(servicer, self.grpc)
        self.port = self.grpc.add_insecure_port(f"[::]:{port}")

    def start(self) -> "Server":
        self.grpc.start()
        log.info("listening on :%d", self.port)
        return self

    def wait_until_done(self, grace: float = 30.0, tick: float = 0.5) -> None:
        """Drive round timeouts until training is over and clients have been told."""
        while not self.servicer.finished(grace):
            self.servicer.coord.check_timeout()
            time.sleep(tick)

    def stop(self) -> None:
        stopped = self.grpc.stop(grace=2)  # refuse new RPCs; cancel open ones after 2 s
        # Long-polls sit in coord.wait() for up to 20 s, and Python waits for
        # every gRPC worker thread before it exits. Release them now.
        self.servicer.coord.stop()
        stopped.wait()

    def run_until_done(self, grace: float = 30.0, tick: float = 0.5) -> None:
        self.wait_until_done(grace, tick)
        log.info("training finished; shutting down")
        self.stop()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kubesplit server", description=__doc__)
    p.add_argument("--mode", choices=["vanilla", "decoupled"], default=env("MODE", "decoupled"))
    p.add_argument("--clients", type=int, default=env("CLIENTS", "4"))
    p.add_argument("--rounds", type=int, default=env("ROUNDS", "5"))
    p.add_argument("--quorum-pct", type=float, default=env("QUORUM_PCT", "100"))
    p.add_argument("--round-timeout", type=float, default=env("ROUND_TIMEOUT", "600"))
    p.add_argument("--partition", choices=["iid", "dirichlet"], default=env("PARTITION", "iid"))
    p.add_argument("--alpha", type=float, default=env("ALPHA", "0.5"))
    p.add_argument("--seed", type=int, default=env("SEED", "0"))
    p.add_argument("--batch-size", type=int, default=env("BATCH_SIZE", "64"))
    p.add_argument("--lr", type=float, default=env("LR", "0.05"))
    p.add_argument("--dataset", choices=["cifar10", "fake"], default=env("DATASET", "cifar10"))
    p.add_argument("--max-batches", type=int, default=env("MAX_BATCHES", "0"))
    p.add_argument("--port", type=int, default=env("PORT", "50051"))
    p.add_argument("--metrics-port", type=int, default=env("METRICS_PORT", "8000"), help="0 = off")
    p.add_argument("--results-dir", default=env("RESULTS_DIR", "results"))
    p.add_argument("--run-id", default=env("RUN_ID", ""))
    p.add_argument("--done-grace", type=float, default=env("DONE_GRACE", "30"))
    p.add_argument(
        "--exit-on-done",
        type=parse_bool,
        default=env("EXIT_ON_DONE", "true"),
        help="false = stay up after training (Kubernetes would restart an exited server)",
    )
    add_common_args(p)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    setup(a.log_level, a.threads)
    cfg = pb.RunConfig(
        mode=a.mode,
        num_clients=a.clients,
        partition=a.partition,
        alpha=a.alpha,
        seed=a.seed,
        batch_size=a.batch_size,
        lr=a.lr,
        dataset=a.dataset,
        max_batches=a.max_batches,
    )
    run_id = a.run_id or f"{a.mode}-c{a.clients}-{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(a.results_dir, exist_ok=True)
    results_path = os.path.join(a.results_dir, f"{run_id}.csv")

    servicer = SplitLearningServicer(
        cfg,
        rounds=a.rounds,
        test_set=load_dataset(a.data_root, fake=a.dataset == "fake", train=False),
        quorum_pct=a.quorum_pct,
        round_timeout=a.round_timeout,
        results_path=results_path,
        device=device(),
    )
    log.info(
        "run %s (server id %s): %s, %d clients, %d rounds, quorum %d",
        run_id,
        servicer.server_id,
        a.mode,
        a.clients,
        a.rounds,
        servicer.coord.quorum,
    )
    if a.metrics_port:
        log.info("metrics on :%d/metrics", servicer.metrics.serve(a.metrics_port))
    server = Server(servicer, a.port).start()
    try:
        server.wait_until_done(a.done_grace)
        log.info("training finished; results written to %s", results_path)
        if not a.exit_on_done:
            log.info("staying up for metrics scrapes and late clients (KS_EXIT_ON_DONE=false)")
            idle_forever()  # until the pod is deleted (SIGTERM)
    finally:
        server.stop()  # on SIGTERM too, so open RPCs end and the process can exit
    return 0
