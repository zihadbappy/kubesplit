"""Kill a kubesplit pod mid-round and time how training recovers.

    python scripts/pod_kill.py client-2                 # once round 3 is under way
    python scripts/pod_kill.py server --at-round 5 --out results/pod-kill.csv

It watches the run until round --at-round opens (timing the round before it as
a baseline), waits --delay seconds so training is mid-round, deletes the pod,
then reports when each stage of recovery happened, in seconds after the delete:

  pod_gone       the killed pod no longer exists
  pod_running    a replacement container is running
  server_up      (server only) the replacement serves metrics
  rejoined       client: the server counted its new Join
                 server: every client has joined the new run
  round_closed   a round closed with the replacement back: training moves again

Pod state comes from kubectl; progress from the server's own /metrics, read
through the Kubernetes API's service proxy, so neither Prometheus nor a
port-forward is needed. Polling makes times accurate to about a second.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families

POLL = 0.25
STAGES = ["pod_gone", "pod_running", "server_up", "rejoined", "round_closed"]


class Kube:
    def __init__(self, namespace: str, context: str | None):
        self.namespace = namespace
        self.cmd = ["kubectl", "-n", namespace] + (["--context", context] if context else [])

    def run(self, *args: str, check: bool = True) -> str | None:
        r = subprocess.run([*self.cmd, *args], capture_output=True, text=True)
        if r.returncode:
            if check:
                sys.exit(f"kubectl {' '.join(args)}: {r.stderr.strip()}")
            return None
        return r.stdout

    def items(self, kind: str, selector: str) -> list[dict]:
        return json.loads(self.run("get", kind, "-l", selector, "-o", "json"))["items"]


def metrics(kube: Kube, service: str) -> dict[str, float] | None:
    """The server's /metrics, summed over labels. None while it is unreachable."""
    path = f"/api/v1/namespaces/{kube.namespace}/services/{service}:metrics/proxy/metrics"
    text = kube.run("get", "--raw", path, check=False)
    if text is None:
        return None
    out: dict[str, float] = {}
    for family in text_string_to_metric_families(text):
        for s in family.samples:
            out[s.name] = out.get(s.name, 0.0) + s.value
    return out


def running(pod: dict) -> bool:
    if pod["metadata"].get("deletionTimestamp"):
        return False  # terminating
    statuses = pod.get("status", {}).get("containerStatuses", [])
    return any("running" in cs.get("state", {}) for cs in statuses)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("target", help="server, or client-N (N = the client's shard)")
    p.add_argument("--release", default="kubesplit", help="Helm release name")
    p.add_argument("-n", "--namespace", default="default")
    p.add_argument("--context", help="kubectl context (default: the current one)")
    p.add_argument("--at-round", type=int, default=3, help="kill once this round is open")
    p.add_argument("--delay", type=float, default=1.0, help="seconds into the round to kill")
    p.add_argument("--timeout", type=float, default=900, help="give up after this many seconds")
    p.add_argument("--out", help="append the result to this CSV file")
    a = p.parse_args()
    if a.target != "server" and not a.target.startswith("client-"):
        p.error("target must be server or client-N")
    is_server = a.target == "server"

    kube = Kube(a.namespace, a.context)
    release = f"app.kubernetes.io/instance={a.release}"
    services = kube.items("svc", f"{release},app.kubernetes.io/component=server")
    sets = kube.items("statefulset", f"{release},app.kubernetes.io/component=client")
    deploys = kube.items("deploy", f"{release},app.kubernetes.io/component=server")
    if not (services and sets and deploys):
        sys.exit(f"no kubesplit release {a.release!r} in namespace {a.namespace!r}")
    service = services[0]["metadata"]["name"]
    clients = sets[0]["spec"]["replicas"]
    env = {
        e["name"]: e.get("value")
        for e in deploys[0]["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    mode = env.get("KS_MODE", "?")
    component = "server" if is_server else "client"
    pod_selector = f"{release},app.kubernetes.io/component={component}"
    victim_name = None if is_server else f"{sets[0]['metadata']['name']}-{a.target.split('-')[1]}"

    # Baseline: watch two round openings and time the round between them.
    print(f"{a.release} ({mode}, {clients} clients): waiting for round {a.at_round} to open")
    opened: dict[int, float] = {}
    first = None
    while True:
        m = metrics(kube, service)
        if m is None:
            sys.exit("server metrics unreachable; is the server running?")
        if m.get("kubesplit_training_done"):
            sys.exit("training already finished; start a run with more rounds")
        rnd = int(m["kubesplit_round"])
        first = rnd if first is None else first
        opened.setdefault(rnd, time.monotonic())
        if rnd >= a.at_round and rnd - 1 in opened and rnd - 1 != first:
            round_s = opened[rnd] - opened[rnd - 1]
            break
        time.sleep(POLL)
    time.sleep(a.delay)

    before = metrics(kube, service)
    victims = [
        pod
        for pod in kube.items("pods", pod_selector)
        if running(pod) and (is_server or pod["metadata"]["name"] == victim_name)
    ]
    if len(victims) != 1:
        sys.exit(f"expected one running {a.target} pod, found {len(victims)}")
    victim = victims[0]["metadata"]
    print(
        f"round {rnd - 1} took {round_s:.1f}s; deleting {victim['name']} "
        f"{a.delay:g}s into round {rnd}"
    )
    t0 = time.monotonic()
    kube.run("delete", "pod", victim["name"], "--wait=false")

    stages: dict[str, float] = {}
    resumed_from = 0  # the replacement server's kubesplit_resumed_from_round

    def mark(stage: str) -> None:
        if stage not in stages:
            stages[stage] = round(time.monotonic() - t0, 1)
            print(f"  +{stages[stage]:5.1f}s  {stage}")

    while "round_closed" not in stages and time.monotonic() - t0 < a.timeout:
        pods = kube.items("pods", pod_selector)
        if all(pod["metadata"]["uid"] != victim["uid"] for pod in pods):
            mark("pod_gone")
        if any(
            running(pod)
            and pod["metadata"]["uid"] != victim["uid"]
            and (is_server or pod["metadata"]["name"] == victim_name)
            for pod in pods
        ):
            mark("pod_running")
        m = metrics(kube, service)
        if m is not None and is_server:
            # A new server process: new start time, counters back at zero.
            if m["process_start_time_seconds"] != before["process_start_time_seconds"]:
                mark("server_up")
                resumed_from = int(m.get("kubesplit_resumed_from_round", 0))
                if m["kubesplit_clients_joined"] >= clients:
                    mark("rejoined")
                if m["kubesplit_rounds_total"] >= 1 and "rejoined" in stages:
                    mark("round_closed")
        elif m is not None:
            if m["kubesplit_client_joins_total"] > before["kubesplit_client_joins_total"]:
                mark("rejoined")
            if m["kubesplit_rounds_total"] > before["kubesplit_rounds_total"]:
                if "rejoined" in stages:
                    mark("round_closed")
        time.sleep(POLL)

    recovered = "round_closed" in stages
    # Rounds closed when the server died, minus those a checkpoint brought back.
    lost = max(0, int(before["kubesplit_round"]) - 1 - resumed_from) if is_server else 0
    if not is_server:
        outcome = ""
    elif resumed_from:
        outcome = f"; the new server resumed after round {resumed_from}, losing {lost} rounds"
    else:
        outcome = f"; the new run restarted from round 1, losing {lost} rounds"
    print(
        f"{'recovered' if recovered else 'NOT recovered'}: training moving again "
        f"{stages.get('round_closed', float('nan')):.1f}s after the kill "
        f"(a normal round took {round_s:.1f}s){outcome}"
    )

    if a.out:
        row = {
            "utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "release": a.release,
            "mode": mode,
            "clients": clients,
            "target": a.target,
            "killed_in_round": rnd,
            "normal_round_s": round(round_s, 1),
            **{s: stages.get(s, "") for s in STAGES},
            "rounds_lost": lost,
        }
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        new = not out.exists()
        with out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"appended to {out}")
    return 0 if recovered else 1


if __name__ == "__main__":
    sys.exit(main())
