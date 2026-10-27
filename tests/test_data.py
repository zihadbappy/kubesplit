import numpy as np
import pytest

from kubesplit.data import dataset_labels, dirichlet_partition, fake_cifar10, iid_partition

LABELS = np.repeat(np.arange(10), 100)  # 1000 samples, 100 per class


def assert_exact_cover(shards, n):
    allidx = np.concatenate(shards)
    assert len(allidx) == n
    assert len(np.unique(allidx)) == n  # disjoint and complete


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_iid_is_disjoint_complete_and_balanced(k):
    shards = iid_partition(LABELS, k, seed=0)
    assert_exact_cover(shards, len(LABELS))
    for s in shards:
        counts = np.bincount(LABELS[s], minlength=10)
        assert counts.max() - counts.min() <= 1  # every class equally represented


@pytest.mark.parametrize("k", [2, 4, 8])
def test_dirichlet_is_disjoint_complete_and_nonempty(k):
    shards = dirichlet_partition(LABELS, k, alpha=0.5, seed=0)
    assert_exact_cover(shards, len(LABELS))
    assert min(len(s) for s in shards) >= 10


def test_dirichlet_is_skewed():
    shards = dirichlet_partition(LABELS, 4, alpha=0.5, seed=0)
    iid = iid_partition(LABELS, 4, seed=0)

    def label_spread(parts):  # mean per-client std of the class histogram
        return np.mean([np.bincount(LABELS[s], minlength=10).std() for s in parts])

    assert label_spread(shards) > 3 * label_spread(iid)


@pytest.mark.parametrize("fn", [iid_partition, dirichlet_partition])
def test_partition_is_deterministic(fn):
    # Every pod recomputes the split on its own, so it must not depend on global RNG.
    a = fn(LABELS, 4, seed=7)
    np.random.seed(123)
    b = fn(LABELS, 4, seed=7)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    c = fn(LABELS, 4, seed=8)
    assert any(not np.array_equal(x, y) for x, y in zip(a, c, strict=True))


def test_fake_data_shape_and_labels():
    ds = fake_cifar10(50)
    x, _ = ds[0]
    assert x.shape == (3, 32, 32)
    assert dataset_labels(ds).shape == (50,)
