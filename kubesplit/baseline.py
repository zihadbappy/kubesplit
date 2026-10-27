"""Single-process baseline: N clients + 1 server, both modes, no network.

Each round every client starts from the global bottom, trains one local
epoch on its shard (talking to the shared server top), then the bottoms are
FedAvg'd weighted by shard size. This is the reference the distributed
version must match.

    python -m kubesplit.baseline --mode both --clients 4 --rounds 5
"""

import argparse
import copy
import time
from dataclasses import asdict, dataclass

import torch
from torch.utils.data import DataLoader

from kubesplit.data import client_shards, load_cifar10
from kubesplit.split import ClientWorker, Mode, ServerWorker, evaluate, fedavg


@dataclass
class RoundStats:
    mode: str
    round: int
    test_acc: float
    train_loss: float
    train_acc: float
    bytes_up: int  # client -> server: activations + labels
    bytes_down: int  # server -> client: gradients (vanilla only)
    seconds: float


def run(
    mode: Mode,
    clients: int = 4,
    rounds: int = 5,
    partition: str = "iid",
    alpha: float = 0.5,
    batch_size: int = 64,
    lr: float = 0.05,
    seed: int = 0,
    data_root: str = "./data",
    fake_data: bool = False,
    max_batches: int | None = None,
    device: str | None = None,
    verbose: bool = True,
) -> list[RoundStats]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    train, test = load_cifar10(data_root, fake=fake_data)
    shards = client_shards(train, clients, partition, alpha, seed)
    g = torch.Generator().manual_seed(seed)
    loaders = [DataLoader(s, batch_size=batch_size, shuffle=True, generator=g) for s in shards]
    test_loader = DataLoader(test, batch_size=256)

    workers = [ClientWorker(mode, lr, device) for _ in range(clients)]
    for w in workers[1:]:  # all clients start from the same weights
        w.modules().load_state_dict(workers[0].modules().state_dict())
    global_state = copy.deepcopy(workers[0].modules().state_dict())
    server = ServerWorker(lr, device)

    history = []
    for r in range(1, rounds + 1):
        t0 = time.perf_counter()
        loss_sum = correct = seen = up = down = 0
        for w, loader in zip(workers, loaders, strict=True):
            w.modules().load_state_dict(global_state)
            w.train()
            for b, (x, y) in enumerate(loader):
                if max_batches is not None and b >= max_batches:
                    break
                if mode == "vanilla":
                    h = w.forward(x)
                    res = server.step(h, y, want_grad=True)
                    w.backward(res.grad)
                    down += res.grad.nbytes
                else:
                    h, _ = w.local_step(x, y)
                    res = server.step(h, y, want_grad=False)
                up += h.nbytes + y.nbytes
                loss_sum += res.loss * y.numel()
                correct += res.correct
                seen += y.numel()

        global_state = fedavg([w.modules().state_dict() for w in workers], [len(s) for s in shards])
        bottom = workers[0].bottom
        workers[0].modules().load_state_dict(global_state)
        stats = RoundStats(
            mode=mode,
            round=r,
            test_acc=evaluate(bottom, server.top, test_loader, device),
            train_loss=loss_sum / seen,
            train_acc=100.0 * correct / seen,
            bytes_up=up,
            bytes_down=down,
            seconds=time.perf_counter() - t0,
        )
        history.append(stats)
        if verbose:
            print(
                f"[{mode:9s}] round {r:3d} | test {stats.test_acc:5.1f}% | "
                f"train {stats.train_acc:5.1f}% loss {stats.train_loss:.3f} | "
                f"up {up / 2**20:7.1f} MiB down {down / 2**20:7.1f} MiB | {stats.seconds:6.1f}s",
                flush=True,
            )
    return history


def summarize(results: dict[str, list[RoundStats]]) -> None:
    print("\nmode       final_test_acc  total_MiB  MiB/round  s/round")
    for mode, hist in results.items():
        mib = sum(s.bytes_up + s.bytes_down for s in hist) / 2**20
        secs = sum(s.seconds for s in hist)
        print(
            f"{mode:9s}  {hist[-1].test_acc:13.1f}%  {mib:9.1f}  "
            f"{mib / len(hist):9.1f}  {secs / len(hist):7.1f}"
        )


def main(argv=None) -> dict[str, list[RoundStats]]:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=["vanilla", "decoupled", "both"], default="both")
    p.add_argument("--clients", type=int, default=4)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--partition", choices=["iid", "dirichlet"], default="iid")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--fake-data", action="store_true", help="synthetic data, no download")
    p.add_argument("--max-batches", type=int, default=None, help="cap batches per client per round")
    p.add_argument("--csv", default=None, help="write per-round stats here")
    a = p.parse_args(argv)

    modes = ["vanilla", "decoupled"] if a.mode == "both" else [a.mode]
    results = {
        m: run(
            m,
            clients=a.clients,
            rounds=a.rounds,
            partition=a.partition,
            alpha=a.alpha,
            batch_size=a.batch_size,
            lr=a.lr,
            seed=a.seed,
            data_root=a.data_root,
            fake_data=a.fake_data,
            max_batches=a.max_batches,
        )
        for m in modes
    }
    summarize(results)

    if a.csv:
        import csv

        rows = [asdict(s) for hist in results.values() for s in hist]
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
    return results


if __name__ == "__main__":
    main()
