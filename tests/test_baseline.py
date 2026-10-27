import pytest

from kubesplit.baseline import main, run


@pytest.mark.parametrize("mode", ["vanilla", "decoupled"])
@pytest.mark.parametrize("partition", ["iid", "dirichlet"])
def test_baseline_learns_on_fake_data(mode, partition):
    hist = run(mode, clients=3, rounds=3, partition=partition, fake_data=True, verbose=False)
    assert len(hist) == 3
    assert hist[-1].test_acc > 30  # chance is 10%
    assert hist[-1].train_loss < hist[0].train_loss


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
