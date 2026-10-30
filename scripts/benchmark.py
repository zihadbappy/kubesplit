"""Run the benchmark matrix: a fresh Helm install per run, results to CSV.

    python scripts/benchmark.py --modes vanilla,decoupled --delays 0,50 --repeats 3 \
        --rounds 5 --out benchmarks/gke -- -f deploy/gke/values.yaml \
        --set image.repository=REPO --set image.tag=TAG

Everything after -- goes to helm install. For each run it uninstalls the
release and waits for its pods to go, installs with that mode and
run.netDelayMs, waits until the server reports training done, then writes:

  runs/<run_id>.csv  the server's per-round results (accuracy, bytes, round time)
  summary.csv        one row per run: round times, accuracy, traffic, and the
                     per-batch compute / blocked split averaged over clients

Runs are interleaved (repeat 1 of every config, then repeat 2, ...) so slow
drift in the cloud doesn't land on one config. A fresh install per run
matters: a helm upgrade lets old client pods join the new run before they
are replaced, which inflates its first rounds.
"""

import argparse
import csv
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from pod_kill import Kube  # same directory
from prometheus_client.parser import text_string_to_metric_families

CHART = str(Path(__file__).resolve().parent.parent / "charts" / "kubesplit")
POLL = 5.0


class RaisingKube(Kube):
    """pod_kill's Kube exits the process when kubectl fails; here one failed
    call should fail one run, not the whole matrix."""

    def run(self, *args: str, check: bool = True) -> str | None:
        out = super().run(*args, check=False)
        if out is None and check:
            raise RuntimeError(f"kubectl {' '.join(args[:3])} failed")
        return out


def metrics_port(pod: dict) -> int:
    for c in pod["spec"]["containers"]:
        for port in c.get("ports", []):
            if port.get("name") == "metrics":
                return port["containerPort"]
    raise RuntimeError(f"{pod['metadata']['name']} has no metrics port")


def scrape(kube: Kube, path: str) -> dict[tuple, float]:
    """Samples from a /metrics page behind the API server proxy, keyed by (name, labels)."""
    text = kube.run("get", "--raw", path, check=False)
    out: dict[tuple, float] = {}
    for family in text_string_to_metric_families(text or ""):
        for s in family.samples:
            out[(s.name, tuple(sorted(s.labels.items())))] = s.value
    return out


def total(m: dict[tuple, float], name: str, **labels: str) -> float:
    """Sum of every sample of `name` whose labels include `labels`."""
    want = set(labels.items())
    return sum(v for (n, lab), v in m.items() if n == name and want <= set(lab))


def per_batch_ms(m: dict[tuple, float], hist: str, **labels: str) -> float:
    n = total(m, f"{hist}_count", **labels)
    return 1000 * total(m, f"{hist}_sum", **labels) / n if n else float("nan")


class Bench:
    def __init__(self, a: argparse.Namespace, helm_args: list[str]):
        self.a = a
        self.kube = RaisingKube(a.namespace, a.context)
        self.helm = ["helm", "-n", a.namespace] + (
            ["--kube-context", a.context] if a.context else []
        )
        self.helm_args = helm_args
        self.sel = f"app.kubernetes.io/instance={a.release}"
        self.out = Path(a.out)
        (self.out / "runs").mkdir(parents=True, exist_ok=True)

    def sh(self, *args: str) -> None:
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"{' '.join(args[:4])}...: {r.stderr.strip()[-500:]}")

    def pods(self) -> list[dict]:
        return self.kube.items("pods", self.sel)

    def uninstall(self) -> None:
        self.sh(*self.helm, "uninstall", self.a.release, "--wait", "--ignore-not-found")
        deadline = time.monotonic() + 300
        while self.pods():
            if time.monotonic() > deadline:
                raise RuntimeError("old pods still terminating after 5 min")
            time.sleep(2)

    def install(self, mode: str, delay: int, run_id: str) -> None:
        self.sh(
            *self.helm, "install", self.a.release, CHART, *self.helm_args,
            "--set-string", f"run.mode={mode}",
            "--set", f"run.netDelayMs={delay}",
            "--set", f"run.rounds={self.a.rounds}",
            "--set-string", f"run.runId={run_id}",
            "--wait", "--timeout", "20m",
        )  # fmt: skip

    def server(self) -> tuple[str, str]:
        role = f"{self.sel},app.kubernetes.io/component=server"
        svc = self.kube.items("svc", role)
        pods = [
            p for p in self.kube.items("pods", role) if not p["metadata"].get("deletionTimestamp")
        ]
        return svc[0]["metadata"]["name"], pods[0]["metadata"]["name"]

    def run(self, mode: str, delay: int, repeat: int) -> dict:
        run_id = f"{mode}-d{delay}-r{repeat}-{datetime.now(UTC):%m%d%H%M%S}"
        print(f"[{datetime.now():%H:%M:%S}] {run_id}: reinstalling", flush=True)
        self.uninstall()
        self.install(mode, delay, run_id)
        t0 = time.monotonic()
        service, server_pod = self.server()
        metrics_path = (
            f"/api/v1/namespaces/{self.a.namespace}/services/{service}:metrics/proxy/metrics"
        )
        while True:
            m = scrape(self.kube, metrics_path)
            if total(m, "kubesplit_training_done") == 1:
                break
            if time.monotonic() - t0 > self.a.run_timeout:
                raise RuntimeError(f"not done after {self.a.run_timeout:.0f}s")
            time.sleep(POLL)
        wall = time.monotonic() - t0

        rows_text = self.kube.run(
            "exec", server_pod, "-c", "server", "--", "cat", f"/results/{run_id}.csv"
        )
        (self.out / "runs" / f"{run_id}.csv").write_text(rows_text, newline="")
        rows = list(csv.DictReader(rows_text.splitlines()))
        secs = [float(r["round_seconds"]) for r in rows]

        clients = [
            p
            for p in self.pods()
            if p["metadata"]["labels"].get("app.kubernetes.io/component") == "client"
        ]
        ns = self.a.namespace
        cm = [
            scrape(
                self.kube,
                f"/api/v1/namespaces/{ns}/pods/{p['metadata']['name']}:{metrics_port(p)}/proxy/metrics",
            )
            for p in clients
        ]
        statuses = [cs for p in self.pods() for cs in p["status"].get("containerStatuses", [])]
        restarts = sum(cs.get("restartCount", 0) for cs in statuses)
        joins = total(m, "kubesplit_client_joins_total")
        mib = 1 << 20

        def client_ms(phase: str) -> float:
            each = [per_batch_ms(c, "kubesplit_client_batch_seconds", phase=phase) for c in cm]
            return round(statistics.mean(each), 1)

        row = {
            "utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "run_id": run_id,
            "mode": mode,
            "net_delay_ms": delay,
            "repeat": repeat,
            "clients": len(clients),
            "rounds": len(rows),
            # Clean: no pod restarted and every client joined exactly once.
            "clean": restarts == 0 and joins == len(clients),
            "wall_s": round(wall, 1),
            "round1_s": round(secs[0], 1),
            "steady_round_s": round(statistics.mean(secs[1:]), 2) if len(secs) > 1 else "",
            "final_test_acc": float(rows[-1]["test_acc"]),
            "mib_up": round(sum(int(r["bytes_up"]) for r in rows) / mib, 1),
            "mib_down": round(sum(int(r["bytes_down"]) for r in rows) / mib, 1),
            "mib_sync": round(sum(int(r["bytes_sync"]) for r in rows) / mib, 1),
            "client_compute_ms": client_ms("compute"),
            "client_blocked_ms": client_ms("blocked"),
            "server_step_ms": round(per_batch_ms(m, "kubesplit_server_step_seconds"), 1),
            "server_queue_ms": round(per_batch_ms(m, "kubesplit_server_lock_wait_seconds"), 1),
        }
        note = "" if row["clean"] else f" (NOT clean: {restarts} restarts, {joins:.0f} joins)"
        print(
            f"    done in {wall:.0f}s: round 1 {row['round1_s']}s, then "
            f"{row['steady_round_s']}s/round, acc {row['final_test_acc']}%, "
            f"blocked {row['client_blocked_ms']} ms/batch{note}",
            flush=True,
        )
        return row

    def append(self, row: dict) -> None:
        path = self.out / "summary.csv"
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)


def main() -> int:
    argv = sys.argv[1:]
    helm_args = argv[argv.index("--") + 1 :] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--modes", default="vanilla,decoupled")
    p.add_argument("--delays", default="0,50", help="run.netDelayMs values, comma-separated")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--first-repeat", type=int, default=1, help="number of the first repeat (to resume)"
    )
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--release", default="kubesplit")
    p.add_argument("-n", "--namespace", default="default")
    p.add_argument("--context", help="kubectl context (default: the current one)")
    p.add_argument("--run-timeout", type=float, default=3600, help="seconds per run")
    p.add_argument(
        "--max-failures", type=int, default=2, help="stop after this many failures in a row"
    )
    p.add_argument("--out", default="benchmarks/results")
    a = p.parse_args(argv)

    bench = Bench(a, helm_args)
    modes = a.modes.split(",")
    delays = [int(d) for d in a.delays.split(",")]
    failures = 0
    for repeat in range(a.first_repeat, a.first_repeat + a.repeats):
        for delay in delays:
            for mode in modes:
                try:
                    bench.append(bench.run(mode, delay, repeat))
                    failures = 0
                except Exception as e:  # keep going: one bad run shouldn't sink the matrix
                    failures += 1
                    print(f"    FAILED: {e}", flush=True)
                    if failures >= a.max_failures:
                        print("too many failures in a row; stopping", flush=True)
                        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
