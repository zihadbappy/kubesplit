"""Real gRPC over localhost: one server, client threads, synthetic data."""

import csv
import threading

import grpc
import pytest

from kubesplit import kubesplit_pb2 as pb
from kubesplit import kubesplit_pb2_grpc as pb_grpc
from kubesplit.client import run_client
from kubesplit.data import load_dataset
from kubesplit.server import Server, SplitLearningServicer
from kubesplit.tensors import to_proto


@pytest.fixture
def start_server():
    started = []

    def start(mode, clients=3, rounds=2, quorum_pct=100.0, round_timeout=60.0, results_path=None):
        cfg = pb.RunConfig(
            mode=mode,
            num_clients=clients,
            partition="iid",
            alpha=0.5,
            seed=0,
            batch_size=32,
            lr=0.05,
            dataset="fake",
            max_batches=3,
        )
        servicer = SplitLearningServicer(
            cfg,
            rounds=rounds,
            test_set=load_dataset(fake=True, train=False),
            quorum_pct=quorum_pct,
            round_timeout=round_timeout,
            results_path=results_path,
            poll_seconds=1.0,
        )
        server = Server(servicer, port=0).start()
        t = threading.Thread(
            target=server.run_until_done, kwargs={"grace": 5, "tick": 0.05}, daemon=True
        )
        t.start()
        started.append(server)
        return servicer, server, t

    yield start
    for s in started:
        s.grpc.stop(grace=0)


def run_clients(port, shards):
    trained = {}

    def go(shard):
        trained[shard] = run_client(
            f"localhost:{port}", f"client-{shard}", shard, connect_timeout=30
        )

    threads = [threading.Thread(target=go, args=(s,)) for s in shards]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    return trained


@pytest.mark.parametrize("mode", ["vanilla", "decoupled"])
def test_full_run(start_server, tmp_path, mode):
    out = tmp_path / "run.csv"
    servicer, server, loop = start_server(mode, clients=3, rounds=2, results_path=str(out))
    assert run_clients(server.port, [0, 1, 2]) == {0: 2, 1: 2, 2: 2}
    loop.join(timeout=30)
    assert not loop.is_alive(), "server should exit once every client is told done"

    rows = list(csv.DictReader(out.open()))
    assert [int(r["round"]) for r in rows] == [1, 2]
    for r in rows:
        assert r["mode"] == mode and r["close_reason"] == "quorum" and r["reporting"] == "3"
        assert int(r["bytes_up"]) > 3 * 3 * 32 * 64 * 8 * 8 * 4  # 3 clients x 3 batches
        assert int(r["bytes_sync"]) > 0
        assert (int(r["bytes_down"]) > 0) == (mode == "vanilla")


def test_quorum_lets_rounds_close_without_a_missing_client(start_server):
    servicer, server, loop = start_server("decoupled", clients=3, quorum_pct=60)
    assert run_clients(server.port, [0, 1]) == {0: 2, 1: 2}
    assert [r["close_reason"] for r in servicer.rows] == ["quorum", "quorum"]
    assert [r["reporting"] for r in servicer.rows] == [2, 2]


def test_timeout_closes_rounds_without_a_missing_client(start_server):
    servicer, server, loop = start_server("vanilla", clients=3, round_timeout=1.0)
    assert run_clients(server.port, [0, 1]) == {0: 2, 1: 2}
    assert [r["close_reason"] for r in servicer.rows] == ["timeout", "timeout"]
    assert [r["reporting"] for r in servicer.rows] == [2, 2]


def test_rejects_bad_shard_and_wrong_mode_rpc(start_server):
    _, server, _ = start_server("decoupled", clients=2)
    with grpc.insecure_channel(f"localhost:{server.port}") as ch:
        stub = pb_grpc.SplitLearningStub(ch)
        with pytest.raises(grpc.RpcError) as e:
            stub.Join(pb.JoinRequest(client_id="x", shard=5), timeout=10)
        assert e.value.code() == grpc.StatusCode.INVALID_ARGUMENT

        import torch

        msg = pb.Activations(
            activations=to_proto(torch.zeros(1, 64, 8, 8)), labels=to_proto(torch.zeros(1).long())
        )
        with pytest.raises(grpc.RpcError) as e:
            stub.Step(msg, timeout=10)
        assert e.value.code() == grpc.StatusCode.FAILED_PRECONDITION
