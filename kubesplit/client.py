"""gRPC client: owns the bottom model and one data shard.

    python -m kubesplit client --server localhost:50051 --shard 0

Mode, partitioning and hyperparameters come from the server on Join, so every
client agrees on them. The shard defaults to the hostname's trailing ordinal
(StatefulSet pod kubesplit-client-3 -> shard 3). If the server restarts, the
client joins its new run instead of exiting.
"""

import argparse
import functools
import itertools
import logging
import socket
import time
from dataclasses import dataclass

import grpc
import torch
from torch.utils.data import DataLoader

from kubesplit import kubesplit_pb2 as pb
from kubesplit import kubesplit_pb2_grpc as pb_grpc
from kubesplit.data import client_shards, load_dataset, train_loader
from kubesplit.metrics import ClientMetrics
from kubesplit.runtime import (
    CLIENT_GRPC_OPTIONS,
    LONG_POLL_SECONDS,
    add_common_args,
    device,
    env,
    idle_forever,
    parse_bool,
    setup,
    shard_from_hostname,
)
from kubesplit.split import ClientWorker
from kubesplit.tensors import from_proto, state_from_proto, state_to_proto, to_proto

log = logging.getLogger("kubesplit.client")

STEP_TIMEOUT = 120.0


def _vanilla_round(stub, worker, batches, client_id, rnd, m) -> tuple[int, float, int]:
    """One gradient round trip per batch: the client idles while it waits."""
    compute, blocked = m.batch_seconds.labels("compute"), m.batch_seconds.labels("blocked")
    n = correct = 0
    loss_sum = 0.0
    for x, y in batches:
        t0 = time.perf_counter()
        h = worker.forward(x)
        msg = pb.Activations(
            client_id=client_id, round=rnd, activations=to_proto(h), labels=to_proto(y)
        )
        t1 = time.perf_counter()
        g = stub.Step(msg, timeout=STEP_TIMEOUT)
        t2 = time.perf_counter()
        worker.backward(from_proto(g.grad))
        compute.observe((t1 - t0) + (time.perf_counter() - t2))
        blocked.observe(t2 - t1)
        m.samples.inc(y.numel())
        n += y.numel()
        loss_sum += g.loss * y.numel()
        correct += g.correct
    return n, loss_sum, correct


def _decoupled_round(stub, worker, batches, client_id, rnd, m) -> tuple[int, float, int]:
    """Train locally and stream activations; nothing comes back per batch."""
    compute, blocked = m.batch_seconds.labels("compute"), m.batch_seconds.labels("blocked")
    error: list[BaseException] = []

    def requests():
        try:
            for x, y in batches:
                t0 = time.perf_counter()
                h, _ = worker.local_step(x, y)
                msg = pb.Activations(
                    client_id=client_id, round=rnd, activations=to_proto(h), labels=to_proto(y)
                )
                t1 = time.perf_counter()
                compute.observe(t1 - t0)
                yield msg
                # gRPC asks for the next message once it has room to send: any time
                # spent here is backpressure from the network or a busy server.
                blocked.observe(time.perf_counter() - t1)
                m.samples.inc(y.numel())
        except BaseException as e:  # gRPC swallows iterator errors; re-raise below
            error.append(e)
            raise

    try:
        s = stub.StreamActivations(requests())
    except grpc.RpcError:
        if error:
            raise error[0] from None
        raise
    return s.samples, s.loss_sum, s.correct


@dataclass
class _Session:
    """Everything a client sets up on Join. Rebuilt on every rejoin, so a new
    run starts from a fresh model and optimizer."""

    server_id: str
    cfg: pb.RunConfig
    loader: DataLoader
    worker: ClientWorker
    trained: int = 0


class ServerRestarted(Exception):
    """The server answering is not the one we joined: it restarted, a new run."""


# How a call fails when the server pod goes away mid-call.
SERVER_LOST = (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.CANCELLED)


@functools.lru_cache(maxsize=1)
def _training_set(data_root: str, fake: bool):
    """The whole training split, loaded once per process and reused on rejoin.
    Reloading CIFAR-10 took ~1 s, and held a second 146 MiB copy while the old
    session was still alive. Only the shard split (cheap) depends on the run."""
    return load_dataset(data_root, fake=fake, train=True)


def _join(stub, server, client_id, shard, data_root, device, connect_timeout) -> _Session:
    join = pb.JoinRequest(client_id=client_id, shard=shard)
    reply = stub.Join(join, wait_for_ready=True, timeout=connect_timeout)
    cfg = reply.config
    log.info(
        "%s joined %s as shard %d/%d (%s, server id %s)",
        client_id,
        server,
        shard,
        cfg.num_clients,
        cfg.mode,
        reply.server_id,
    )
    train = _training_set(data_root, cfg.dataset == "fake")
    data = client_shards(train, cfg.num_clients, cfg.partition, cfg.alpha, cfg.seed)[shard]
    g = torch.Generator().manual_seed(cfg.seed * 1000 + shard)
    loader = train_loader(data, cfg.batch_size, g)
    return _Session(reply.server_id, cfg, loader, ClientWorker(cfg.mode, cfg.lr, device))


def _train(stub, s: _Session, client_id: str, connect_timeout: float, m: ClientMetrics) -> None:
    """Train rounds until the server says done. Raises ServerRestarted on a new run."""
    train_round = _vanilla_round if s.cfg.mode == "vanilla" else _decoupled_round
    last = 0
    idle_since = time.perf_counter()
    while True:
        info = stub.WaitRound(
            pb.RoundRequest(client_id=client_id, last_round=last),
            wait_for_ready=True,
            timeout=LONG_POLL_SECONDS + connect_timeout,
        )
        if info.server_id != s.server_id:
            raise ServerRestarted
        if info.done:
            return
        if info.round == 0:  # long-poll expired, nothing new
            continue
        m.idle_seconds.observe(time.perf_counter() - idle_since)
        m.round.set(info.round)

        s.worker.modules().load_state_dict(state_from_proto(info.global_state))
        s.worker.train()
        t0 = time.perf_counter()
        batches = itertools.islice(s.loader, s.cfg.max_batches or None)
        n, loss_sum, correct = train_round(stub, s.worker, batches, client_id, info.round, m)
        m.round_seconds.observe(time.perf_counter() - t0)
        update = pb.Update(
            client_id=client_id,
            round=info.round,
            num_samples=n,
            state=state_to_proto(s.worker.modules().state_dict()),
        )
        ack = stub.SubmitUpdate(update, timeout=STEP_TIMEOUT)
        log.info(
            "%s round %d: %d samples | server loss %.3f acc %.1f%% | %.1fs%s",
            client_id,
            info.round,
            n,
            loss_sum / max(n, 1),
            100.0 * correct / max(n, 1),
            time.perf_counter() - t0,
            "" if ack.accepted else " | late, update discarded",
        )
        last = info.round
        s.trained += 1
        idle_since = time.perf_counter()


def run_client(
    server: str,
    client_id: str,
    shard: int,
    data_root: str = "./data",
    device: str = "cpu",
    connect_timeout: float = 300.0,
    metrics: ClientMetrics | None = None,
) -> int:
    """Train until the server says done. Returns the number of rounds trained.

    If the server restarts or drops mid-round, join again and train its new run
    instead of exiting: Kubernetes would restart the pod anyway, but slower and
    with a growing back-off. Only a server gone for longer than connect_timeout
    ends the process.
    """
    m = metrics or ClientMetrics()
    trained = 0
    with grpc.insecure_channel(server, options=CLIENT_GRPC_OPTIONS) as channel:
        stub = pb_grpc.SplitLearningStub(channel)
        while True:
            s = _join(stub, server, client_id, shard, data_root, device, connect_timeout)
            try:
                _train(stub, s, client_id, connect_timeout, m)
                break
            except ServerRestarted:
                log.warning("%s: the server restarted; joining its new run", client_id)
            except grpc.RpcError as e:
                if e.code() not in SERVER_LOST:
                    raise
                log.warning("%s: lost the server (%s); joining again", client_id, e.code().name)
            finally:
                trained += s.trained
    log.info("%s done after %d rounds", client_id, trained)
    return trained


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kubesplit client", description=__doc__)
    p.add_argument("--server", default=env("SERVER", "localhost:50051"))
    p.add_argument("--shard", type=int, default=env("SHARD", "-1"), help="-1 = from hostname")
    p.add_argument("--client-id", default=env("CLIENT_ID", socket.gethostname()))
    p.add_argument("--connect-timeout", type=float, default=env("CONNECT_TIMEOUT", "300"))
    p.add_argument("--metrics-port", type=int, default=env("METRICS_PORT", "8000"), help="0 = off")
    p.add_argument(
        "--exit-on-done",
        type=parse_bool,
        default=env("EXIT_ON_DONE", "true"),
        help="false = stay up after training (Kubernetes would restart an exited client)",
    )
    add_common_args(p)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    setup(a.log_level, a.threads)
    shard = a.shard if a.shard >= 0 else shard_from_hostname()
    if shard is None:
        raise SystemExit("no --shard / KS_SHARD and hostname has no trailing ordinal")
    metrics = ClientMetrics()
    if a.metrics_port:
        log.info("metrics on :%d/metrics", metrics.serve(a.metrics_port))
    run_client(a.server, a.client_id, shard, a.data_root, device(), a.connect_timeout, metrics)
    if not a.exit_on_done:
        log.info("staying up for metrics scrapes (KS_EXIT_ON_DONE=false)")
        idle_forever()  # until the pod is deleted (SIGTERM)
    return 0
