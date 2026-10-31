"""Real gRPC over localhost: one server, client threads, synthetic data."""

import csv
import threading
import time

import grpc
import pytest

from kubesplit import checkpoint, client
from kubesplit import kubesplit_pb2 as pb
from kubesplit import kubesplit_pb2_grpc as pb_grpc
from kubesplit.client import run_client
from kubesplit.data import load_dataset
from kubesplit.metrics import ClientMetrics
from kubesplit.server import Server, SplitLearningServicer
from kubesplit.tensors import to_proto


@pytest.fixture
def start_server():
    started = []

    def start(
        mode,
        clients=3,
        rounds=2,
        quorum_pct=100.0,
        round_timeout=60.0,
        results_path=None,
        exit_on_done=True,
        port=0,
        **servicer_kwargs,  # checkpoint_path, resume, ...
    ):
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
            **servicer_kwargs,
        )
        server = Server(servicer, port=port).start()
        loop = server.run_until_done if exit_on_done else server.wait_until_done
        t = threading.Thread(target=loop, kwargs={"grace": 5, "tick": 0.05}, daemon=True)
        t.start()
        started.append(server)
        return servicer, server, t

    yield start
    for s in started:
        s.grpc.stop(grace=0)


def run_clients(port, shards, metrics=None):
    trained = {}

    def go(shard):
        trained[shard] = run_client(
            f"localhost:{port}",
            f"client-{shard}",
            shard,
            connect_timeout=30,
            metrics=metrics[shard] if metrics else None,
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


@pytest.mark.parametrize("mode", ["vanilla", "decoupled"])
def test_metrics_agree_with_results(start_server, mode):
    servicer, server, loop = start_server(mode, clients=3, rounds=2)
    clients = {s: ClientMetrics() for s in range(3)}
    assert run_clients(server.port, [0, 1, 2], clients) == {0: 2, 1: 2, 2: 2}
    loop.join(timeout=30)

    def srv(name, **labels):
        return servicer.metrics.registry.get_sample_value(name, labels)

    rows = servicer.rows
    # Traffic counters equal the per-round CSV columns summed.
    acts = srv("kubesplit_transferred_bytes_total", kind="activations")
    grads = srv("kubesplit_transferred_bytes_total", kind="gradients")
    assert acts == sum(r["bytes_up"] for r in rows)
    assert grads == sum(r["bytes_down"] for r in rows)
    assert srv("kubesplit_transferred_bytes_total", kind="weights") == sum(
        r["bytes_sync"] for r in rows
    )
    assert (grads > 0) == (mode == "vanilla")

    batches = 3 * 3 * 2  # clients x max_batches x rounds
    assert srv("kubesplit_server_samples_total") == batches * 32
    assert srv("kubesplit_server_step_seconds_count") == batches
    assert srv("kubesplit_server_lock_wait_seconds_count") == batches
    assert srv("kubesplit_rounds_total", reason="quorum") == 2
    assert srv("kubesplit_rounds_total", reason="timeout") == 0
    assert srv("kubesplit_round_duration_seconds_count") == 2
    assert srv("kubesplit_round_clients_reporting") == 3
    assert srv("kubesplit_round") == 2 and srv("kubesplit_training_done") == 1
    assert srv("kubesplit_client_joins_total") == 3 and srv("kubesplit_clients_joined") == 3
    assert srv("kubesplit_late_updates_total") == 0
    assert srv("kubesplit_test_accuracy_ratio") == pytest.approx(
        rows[-1]["test_acc"] / 100, abs=1e-4
    )

    for m in clients.values():

        def cli(name, **labels):
            return m.registry.get_sample_value(name, labels)  # noqa: B023

        assert cli("kubesplit_client_samples_total") == 2 * 3 * 32
        assert cli("kubesplit_client_batch_seconds_count", phase="compute") == 6
        assert cli("kubesplit_client_batch_seconds_count", phase="blocked") == 6
        assert cli("kubesplit_client_round_seconds_count") == 2
        assert cli("kubesplit_client_idle_seconds_count") == 2
        assert cli("kubesplit_client_round") == 2


def test_server_that_stays_up_tells_a_restarted_client_it_is_done(start_server):
    # In Kubernetes the server keeps running after training (KS_EXIT_ON_DONE=false),
    # and a client pod that restarts afterwards must exit cleanly instead of training.
    servicer, server, loop = start_server("decoupled", clients=2, exit_on_done=False)
    assert run_clients(server.port, [0, 1]) == {0: 2, 1: 2}
    loop.join(timeout=30)
    assert not loop.is_alive() and servicer.coord.done
    assert run_clients(server.port, [0]) == {0: 0}  # rejoins, told done, trains nothing
    assert len(servicer.rows) == 2


@pytest.mark.parametrize("mode", ["vanilla", "decoupled"])
def test_clients_join_a_restarted_server_and_finish_its_run(start_server, mode, monkeypatch):
    # A deleted server pod comes back at the same address as a new server: a new run.
    loads = []
    real_load = client.load_dataset
    monkeypatch.setattr(
        client, "load_dataset", lambda *a, **k: loads.append(1) or real_load(*a, **k)
    )
    client._training_set.cache_clear()

    first, old, _ = start_server(mode, clients=2, rounds=50)
    result = {}
    clients = threading.Thread(target=lambda: result.update(run_clients(old.port, [0, 1])))
    clients.start()
    deadline = time.monotonic() + 60
    while first.coord.round < 3 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert first.coord.round >= 3, "first run never got going"
    loads_before_restart = len(loads)  # 1, or 2 if both threads raced the first load
    old.stop()  # what SIGTERM does

    second, _, loop = start_server(mode, clients=2, rounds=2, port=old.port)
    assert second.server_id != first.server_id
    clients.join(timeout=120)
    assert not clients.is_alive(), "clients should finish the new run, not hang"
    loop.join(timeout=30)
    assert second.coord.done
    assert [r["reporting"] for r in second.rows] == [2, 2]  # both clients came back
    assert all(n >= 2 + 2 for n in result.values())  # rounds in the old run + the new one
    assert len(loads) == loads_before_restart, "rejoining reloaded the training set"


@pytest.mark.parametrize("mode", ["vanilla", "decoupled"])
def test_a_replacement_server_resumes_from_the_checkpoint(start_server, tmp_path, mode):
    # With checkpoints, a replaced server carries the run on instead of starting over.
    results, ckpt = str(tmp_path / "run.csv"), str(tmp_path / "checkpoint.pt")
    run = {"rounds": 4, "results_path": results, "checkpoint_path": ckpt,
           "run_key": "k", "run_id": "r"}  # fmt: skip
    first, old, _ = start_server(mode, clients=2, **run)
    result = {}
    clients = threading.Thread(target=lambda: result.update(run_clients(old.port, [0, 1])))
    clients.start()
    deadline = time.monotonic() + 60
    while first.coord.round < 3 and time.monotonic() < deadline:
        time.sleep(0.05)
    old.stop()  # waits for open RPCs, so no round can close after this
    saved = checkpoint.load(ckpt)
    assert saved["round"] >= 2

    second, _, loop = start_server(mode, clients=2, port=old.port, resume=saved, **run)
    assert second.resumed_from == saved["round"]
    clients.join(timeout=120)
    assert not clients.is_alive()
    loop.join(timeout=30)
    assert second.coord.done
    rounds = [int(r["round"]) for r in csv.DictReader(open(results))]
    assert rounds == [1, 2, 3, 4]  # nothing lost, nothing repeated
    # Each client trained every round; the one in progress at the stop may be redone,
    # since the checkpoint is from the last round that closed.
    assert all(4 <= n <= 5 for n in result.values())


def test_wait_round_answers_a_client_that_never_joined_at_once(start_server):
    # A client from a previous server's run must learn the new server_id right
    # away, not sit in a long-poll for a round number from the old run.
    servicer, server, _ = start_server("decoupled", clients=2)
    with grpc.insecure_channel(f"localhost:{server.port}") as ch:
        stub = pb_grpc.SplitLearningStub(ch)
        joined = stub.Join(pb.JoinRequest(client_id="known", shard=0), timeout=10)
        assert joined.server_id == servicer.server_id
        t0 = time.monotonic()
        info = stub.WaitRound(pb.RoundRequest(client_id="stranger", last_round=7), timeout=10)
        assert time.monotonic() - t0 < 0.5  # a joined client would wait poll_seconds (1 s)
        assert info.round == 0 and info.server_id == servicer.server_id


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
    reg = servicer.metrics.registry
    assert reg.get_sample_value("kubesplit_rounds_total", {"reason": "timeout"}) == 2
    assert reg.get_sample_value("kubesplit_round_clients_reporting") == 2


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
