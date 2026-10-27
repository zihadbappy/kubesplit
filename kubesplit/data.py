"""CIFAR-10 loading and deterministic client sharding.

Partitions depend only on (labels, num_clients, seed[, alpha]), so every
client pod computes the same split independently and keeps the shard at its
own index. No coordination is needed.
"""

from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, TensorDataset

from kubesplit.model import NUM_CLASSES

Partition = Literal["iid", "dirichlet"]

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def iid_partition(labels: np.ndarray, num_clients: int, seed: int = 0) -> list[np.ndarray]:
    """Stratified split: every client gets an equal slice of every class."""
    rng = np.random.default_rng(seed)
    shards: list[list[int]] = [[] for _ in range(num_clients)]
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        for i, part in enumerate(np.array_split(idx, num_clients)):
            shards[i].extend(part.tolist())
    return [np.sort(np.array(s, dtype=np.int64)) for s in shards]


def dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 0,
    min_size: int = 10,
    max_tries: int = 100,
) -> list[np.ndarray]:
    """Non-IID split: per class, proportions across clients ~ Dir(alpha).

    Resamples until every client holds at least `min_size` samples, so no
    pod ends up with an empty shard.
    """
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        shards: list[list[int]] = [[] for _ in range(num_clients)]
        for c in np.unique(labels):
            idx = rng.permutation(np.flatnonzero(labels == c))
            props = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(props)[:-1] * len(idx)).astype(int)
            for i, part in enumerate(np.split(idx, cuts)):
                shards[i].extend(part.tolist())
        if min(len(s) for s in shards) >= min_size:
            return [np.sort(np.array(s, dtype=np.int64)) for s in shards]
    raise ValueError(
        f"no Dirichlet({alpha}) split gave every one of {num_clients} clients "
        f">= {min_size} samples after {max_tries} tries"
    )


def partition(
    labels: np.ndarray,
    num_clients: int,
    kind: Partition = "iid",
    alpha: float = 0.5,
    seed: int = 0,
) -> list[np.ndarray]:
    if kind == "iid":
        return iid_partition(labels, num_clients, seed)
    if kind == "dirichlet":
        return dirichlet_partition(labels, num_clients, alpha, seed)
    raise ValueError(f"unknown partition {kind!r}")


def fake_cifar10(n: int, seed: int = 0) -> TensorDataset:
    """Synthetic CIFAR-shaped data with a learnable class signal.

    Used by tests and CI so nothing has to download the real dataset.
    """
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    prototypes = torch.randn(NUM_CLASSES, 3, 32, 32, generator=torch.Generator().manual_seed(1234))
    x = prototypes[y] + 0.5 * torch.randn(n, 3, 32, 32, generator=g)
    return TensorDataset(x, y)


def load_dataset(root: str = "./data", fake: bool = False, train: bool = True) -> Dataset:
    """One CIFAR-10 split. With `fake`, a small synthetic set instead."""
    if fake:
        return fake_cifar10(2000, seed=0) if train else fake_cifar10(500, seed=1)

    from torchvision import datasets, transforms

    tf = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if train:
        tf = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), *tf]
    return datasets.CIFAR10(root, train=train, download=True, transform=transforms.Compose(tf))


def load_cifar10(root: str = "./data", fake: bool = False) -> tuple[Dataset, Dataset]:
    """Returns (train, test)."""
    return load_dataset(root, fake, train=True), load_dataset(root, fake, train=False)


def dataset_labels(ds: Dataset) -> np.ndarray:
    if isinstance(ds, TensorDataset):
        return ds.tensors[1].numpy()
    return np.asarray(ds.targets)


def client_shards(
    train: Dataset,
    num_clients: int,
    kind: Partition = "iid",
    alpha: float = 0.5,
    seed: int = 0,
) -> list[Subset]:
    parts = partition(dataset_labels(train), num_clients, kind, alpha, seed)
    return [Subset(train, p.tolist()) for p in parts]
