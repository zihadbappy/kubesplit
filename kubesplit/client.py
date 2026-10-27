"""gRPC client: owns the bottom model and one data shard.

    python -m kubesplit client --server localhost:50051 --shard 0

Mode, partitioning and hyperparameters come from the server on Join, so every
client agrees on them. The shard defaults to the hostname's trailing ordinal
(StatefulSet pod kubesplit-client-3 -> shard 3).
"""

import argparse
import itertools
import logging
import socket
import time

import grpc
import torch
from torch.utils.data import DataLoader

from kubesplit import kubesplit_pb2 as pb
from kubesplit import kubesplit_pb2_grpc as pb_grpc
from kubesplit.data import client_shards, load_dataset
from kubesplit.runtime import (
    GRPC_OPTIONS,
    LONG_POLL_SECONDS,
    add_common_args,
    device,
    env,
    setup,
    shard_from_hostname,
)
from kubesplit.split import ClientWorker
from kubesplit.tensors import from_proto, state_from_proto, state_to_proto, to_proto

log = logging.getLogger("kubesplit.client")

STEP_TIMEOUT = 120.0


def _vanilla_round(stub, worker, batches, client_id, rnd) -> tuple[int, float, int]:
    """One gradient round trip per batch: the client idles while it waits."""
    n = correct = 0
    loss_sum = 0.0
    for x, y in batches:
        h = worker.forward(x)
        msg = pb.Activations(
            client_id=client_id, round=rnd, activations=to_proto(h), labels=to_proto(y)
        )
        g = stub.Step(msg, timeout=STEP_TIMEOUT)
        worker.backward(from_proto(g.grad))
        n += y.numel()
        loss_sum += g.loss * y.numel()
        correct += g.correct
    return n, loss_sum, correct


def _decoupled_round(stub, worker, batches, client_id, rnd) -> tuple[int, float, int]:
    """Train locally and stream activations; nothing comes back per batch."""
    error: list[BaseException] = []

    def requests():
        try:
            for x, y in batches:
                h, _ = worker.local_step(x, y)
                yield pb.Activations(
                    client_id=client_id, round=rnd, activations=to_proto(h), labels=to_proto(y)
                )
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


def run_client(
    server: str,
    client_id: str,
    shard: int,
    data_root: str = "./data",
    device: str = "cpu",
    connect_timeout: float = 300.0,
) -> int:
    """Train until the server says done. Returns the number of rounds trained."""
    with grpc.insecure_channel(server, options=GRPC_OPTIONS) as channel:
        stub = pb_grpc.SplitLearningStub(channel)
        join = pb.JoinRequest(client_id=client_id, shard=shard)
        cfg = stub.Join(join, wait_for_ready=True, timeout=connect_timeout).config
        log.info(
            "%s joined %s as shard %d/%d (%s)",
            client_id,
            server,
            shard,
            cfg.num_clients,
            cfg.mode,
        )

        train = load_dataset(data_root, fake=cfg.dataset == "fake", train=True)
        data = client_shards(train, cfg.num_clients, cfg.partition, cfg.alpha, cfg.seed)[shard]
        g = torch.Generator().manual_seed(cfg.seed * 1000 + shard)
        loader = DataLoader(data, batch_size=cfg.batch_size, shuffle=True, generator=g)
        worker = ClientWorker(cfg.mode, cfg.lr, device)
        train_round = _vanilla_round if cfg.mode == "vanilla" else _decoupled_round

        last = trained = 0
        while True:
            info = stub.WaitRound(
                pb.RoundRequest(client_id=client_id, last_round=last),
                wait_for_ready=True,
                timeout=LONG_POLL_SECONDS + connect_timeout,
            )
            if info.done:
                break
            if info.round == 0:  # long-poll expired, nothing new
                continue

            worker.modules().load_state_dict(state_from_proto(info.global_state))
            worker.train()
            t0 = time.perf_counter()
            batches = itertools.islice(loader, cfg.max_batches or None)
            n, loss_sum, correct = train_round(stub, worker, batches, client_id, info.round)
            update = pb.Update(
                client_id=client_id,
                round=info.round,
                num_samples=n,
                state=state_to_proto(worker.modules().state_dict()),
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
            trained += 1
    log.info("%s done after %d rounds", client_id, trained)
    return trained


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kubesplit client", description=__doc__)
    p.add_argument("--server", default=env("SERVER", "localhost:50051"))
    p.add_argument("--shard", type=int, default=env("SHARD", "-1"), help="-1 = from hostname")
    p.add_argument("--client-id", default=env("CLIENT_ID", socket.gethostname()))
    p.add_argument("--connect-timeout", type=float, default=env("CONNECT_TIMEOUT", "300"))
    add_common_args(p)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    setup(a.log_level, a.threads)
    shard = a.shard if a.shard >= 0 else shard_from_hostname()
    if shard is None:
        raise SystemExit("no --shard / KS_SHARD and hostname has no trailing ordinal")
    run_client(a.server, a.client_id, shard, a.data_root, device(), a.connect_timeout)
    return 0
