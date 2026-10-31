import os

import torch

from kubesplit import checkpoint
from kubesplit import kubesplit_pb2 as pb
from kubesplit.data import load_dataset
from kubesplit.server import SplitLearningServicer
from kubesplit.split import ServerWorker


def cfg(**kw) -> pb.RunConfig:
    base = dict(mode="decoupled", num_clients=2, partition="iid", alpha=0.5, seed=0,
                batch_size=32, lr=0.05, dataset="fake", max_batches=3)  # fmt: skip
    return pb.RunConfig(**{**base, **kw})


def test_fingerprint_changes_with_any_run_setting():
    key = checkpoint.fingerprint(cfg(), rounds=5)
    assert key == checkpoint.fingerprint(cfg(), rounds=5)
    assert key != checkpoint.fingerprint(cfg(lr=0.1), rounds=5)
    assert key != checkpoint.fingerprint(cfg(mode="vanilla"), rounds=5)
    assert key != checkpoint.fingerprint(cfg(), rounds=6)
    assert key != checkpoint.fingerprint(cfg(), rounds=5, run_id="bench-1")


def test_save_and_load_round_trip_model_and_optimizer(tmp_path):
    w = ServerWorker()
    w.step(torch.randn(4, 64, 8, 8), torch.tensor([0, 1, 2, 3]), want_grad=False)  # momentum
    path = str(tmp_path / "checkpoint.pt")
    checkpoint.save(path, fingerprint="k", run_id="r", round=3,
                    top=w.top.state_dict(), opt=w.opt.state_dict())  # fmt: skip
    assert not os.path.exists(path + ".tmp")

    got = checkpoint.load(path)
    assert (got["fingerprint"], got["run_id"], got["round"]) == ("k", "r", 3)
    fresh = ServerWorker()
    fresh.top.load_state_dict(got["top"])
    fresh.opt.load_state_dict(got["opt"])
    for a, b in zip(fresh.top.parameters(), w.top.parameters(), strict=True):
        torch.testing.assert_close(a, b)
    assert fresh.opt.state_dict()["state"].keys() == w.opt.state_dict()["state"].keys()


def test_no_checkpoint_means_a_fresh_start(tmp_path):
    assert checkpoint.load(None) is None
    assert checkpoint.load(str(tmp_path / "missing.pt")) is None


def test_a_resumed_server_holds_the_saved_models(tmp_path):
    path = str(tmp_path / "checkpoint.pt")
    first = SplitLearningServicer(cfg(), rounds=4, test_set=load_dataset(fake=True, train=False))
    first.worker.step(torch.randn(4, 64, 8, 8), torch.tensor([0, 1, 2, 3]), want_grad=False)
    state = {
        k: v + 1.0 if v.is_floating_point() else v for k, v in first.coord.global_state.items()
    }
    first.checkpoint_path, first.run_key, first.run_id = path, "k", "r"
    first._save_checkpoint(2, state)

    saved = checkpoint.load(path)
    second = SplitLearningServicer(
        cfg(), rounds=4, test_set=load_dataset(fake=True, train=False), resume=saved
    )
    assert second.coord.round == 3 and not second.coord.done
    for k, v in state.items():
        torch.testing.assert_close(second.coord.global_state[k], v)
    for a, b in zip(second.worker.top.parameters(), first.worker.top.parameters(), strict=True):
        torch.testing.assert_close(a, b)
    reg = second.metrics.registry
    assert reg.get_sample_value("kubesplit_resumed_from_round") == 2
