"""usage: python -m kubesplit {server,client,prepare-data,baseline} [options]

The role may instead come from KS_ROLE, so one image serves every container.
Run a role with --help for its options.
"""

import os
import sys


def prepare_data(argv) -> int:
    """Download CIFAR-10 once so server and clients never race to fetch it."""
    import argparse

    from kubesplit.data import load_dataset
    from kubesplit.runtime import env

    p = argparse.ArgumentParser(prog="kubesplit prepare-data")
    p.add_argument("--data-root", default=env("DATA_ROOT", "./data"))
    p.add_argument("--dataset", choices=["cifar10", "fake"], default=env("DATASET", "cifar10"))
    a = p.parse_args(argv)
    if a.dataset == "fake":
        print("synthetic dataset; nothing to download")
        return 0
    for train in (True, False):
        print(
            f"{'train' if train else 'test'}: {len(load_dataset(a.data_root, train=train))} images"
        )
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    role = argv.pop(0) if argv and not argv[0].startswith("-") else os.environ.get("KS_ROLE", "")
    if role == "server":
        from kubesplit.server import main as run
    elif role == "client":
        from kubesplit.client import main as run
    elif role == "prepare-data":
        run = prepare_data
    elif role == "baseline":
        from kubesplit.baseline import main as baseline

        def run(args):
            baseline(args)
            return 0
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
