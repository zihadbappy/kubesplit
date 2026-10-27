"""Process-level plumbing shared by server and client: env config, CPU limits, gRPC options."""

import logging
import math
import os
import re
import socket

import torch

MiB = 1 << 20
GRPC_OPTIONS = [
    ("grpc.max_send_message_length", 64 * MiB),
    ("grpc.max_receive_message_length", 64 * MiB),
]
# The server holds WaitRound open at most this long; clients allow extra slack.
LONG_POLL_SECONDS = 20.0


def env(name: str, default: str) -> str:
    """KS_<NAME> from the environment. Helm and Compose configure pods this way.

    argparse runs `type` on string defaults, so callers can pass the result
    straight to `default=`.
    """
    return os.environ.get(f"KS_{name}", default)


def shard_from_hostname(hostname: str | None = None) -> int | None:
    """Trailing ordinal of a StatefulSet pod (kubesplit-client-3 -> 3) or Compose host."""
    m = re.search(r"-(\d+)$", hostname or socket.gethostname())
    return int(m.group(1)) if m else None


def cpu_limit() -> int:
    """CPUs this container may actually use.

    os.cpu_count() reports the whole node, so torch would start one thread per
    host core and thrash under a Kubernetes CPU limit. Read the cgroup v2
    quota and the affinity mask instead.
    """
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            n = min(n, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return max(1, n or 1)


def setup(log_level: str = "INFO", threads: int = 0) -> None:
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    torch.set_num_threads(threads or cpu_limit())


def add_common_args(p) -> None:
    p.add_argument("--log-level", default=env("LOG_LEVEL", "INFO"))
    p.add_argument(
        "--threads", type=int, default=env("THREADS", "0"), help="torch threads; 0 = CPU limit"
    )
    p.add_argument("--data-root", default=env("DATA_ROOT", "./data"))


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
