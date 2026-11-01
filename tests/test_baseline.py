import math

import pytest

from kubesplit.baseline import main, run


@pytest.mark.parametrize("mode", ["vanilla", "decoupled"])
def test_baseline_learns_on_fake_data(mode):
    hist = run(mode, clients=3, rounds=3, fake_data=True, verbose=False)
    assert len(hist) == 3
    assert hist[-1].test_acc > 30  # chance is 10%
    assert hist[-1].train_loss < hist[0].train_loss


@pytest.mark.parametrize("mode", ["vanilla", "decoupled"])
def test_baseline_runs_on_skewed_shards(mode):
    # Three FedAvg rounds on Dirichlet shards are too few to learn reliably: over
    # 8 seeds and 3 CPU thread counts final accuracy ranged 16-100% (client drift),
    # so this checks only that the run completes with sane numbers.
    hist = run(mode, clients=3, rounds=3, partition="dirichlet", fake_data=True, verbose=False)
    assert len(hist) == 3
    assert all(0 <= r.test_acc <= 100 and math.isfinite(r.train_loss) for r in hist)


def test_decoupled_trains_a_shard_that_leaves_one_sample_over():
    # The fake set's 2 IID shards hold 1003 and 997 samples. In batches of 167 the
    # first is 6 full batches and 1 sample over, which the aux head's BatchNorm
    # cannot train on.
    hist = run("decoupled", clients=2, rounds=1, fake_data=True, batch_size=167, verbose=False)
    assert len(hist) == 1


def test_bytes_accounting():
    v = run("vanilla", clients=2, rounds=1, fake_data=True, max_batches=2, verbose=False)[0]
    d = run("decoupled", clients=2, rounds=1, fake_data=True, max_batches=2, verbose=False)[0]
    assert d.bytes_down == 0
    assert v.bytes_up == d.bytes_up
    # gradients are the same size as activations; only labels are extra on the way up
    assert v.bytes_up > v.bytes_down > 0.95 * v.bytes_up


def test_cli_writes_csv(tmp_path):
    out = tmp_path / "r.csv"
    main(
        ["--fake-data", "--rounds", "1", "--clients", "2", "--max-batches", "2", "--csv", str(out)]
    )
    lines = out.read_text().splitlines()
    assert lines[0].startswith("mode,round,test_acc")
    assert len(lines) == 3  # header + one round per mode
