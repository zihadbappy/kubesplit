"""Run the server and N clients as separate local processes over real gRPC.

Same code and flags as the containers, without Docker:

    python scripts/local_run.py --mode vanilla --clients 4 --rounds 2 --dataset fake

Exits non-zero if any process fails.
"""

import argparse
import os
import subprocess
import sys


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=["vanilla", "decoupled"], default="decoupled")
    p.add_argument("--clients", type=int, default=4)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--dataset", choices=["cifar10", "fake"], default="cifar10")
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--port", type=int, default=50051)
    p.add_argument("--threads", type=int, default=0, help="torch threads per process; 0 = auto")
    p.add_argument("--results-dir", default="results")
    a, server_extra = p.parse_known_args()

    threads = a.threads or max(1, (os.cpu_count() or 1) // (a.clients + 1))
    py = [sys.executable, "-m", "kubesplit"]
    common = ["--threads", str(threads)]
    server_args = {
        "--mode": a.mode,
        "--clients": a.clients,
        "--rounds": a.rounds,
        "--dataset": a.dataset,
        "--max-batches": a.max_batches,
        "--port": a.port,
        "--results-dir": a.results_dir,
    }
    server_cmd = [*py, "server", *common, *server_extra]
    for flag, value in server_args.items():
        server_cmd += [flag, str(value)]
    server = subprocess.Popen(server_cmd)

    clients = []
    for i in range(a.clients):
        cmd = [*py, "client", *common, "--server", f"localhost:{a.port}", "--shard", str(i)]
        clients.append(subprocess.Popen([*cmd, "--client-id", f"client-{i}"]))
    codes = [c.wait() for c in clients] + [server.wait()]
    return max(codes, key=abs)


if __name__ == "__main__":
    sys.exit(main())
