"""Server checkpoints: carry a run on after the server pod is replaced.

After each round the server saves what it needs to continue: the averaged
client model, its own top model and optimizer, and the round number. A new
server process with the same run settings loads it and opens the next round;
clients rejoin it as they would any new server, so they need no changes.
"""

import hashlib
import os

import torch

from kubesplit import kubesplit_pb2 as pb


def fingerprint(cfg: pb.RunConfig, rounds: int, run_id: str = "") -> str:
    """Identifies a run, so a checkpoint is only resumed by the same settings."""
    key = cfg.SerializeToString(deterministic=True) + f"|{rounds}|{run_id}".encode()
    return hashlib.sha256(key).hexdigest()[:16]


def save(path: str, **state) -> None:
    """Write to a temp file, then rename: a kill mid-write keeps the old checkpoint."""
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load(path: str | None) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    # weights_only: tensors and plain values only, never arbitrary pickled objects.
    return torch.load(path, map_location="cpu", weights_only=True)
