"""SIGTERM handling: Kubernetes stops pods with SIGTERM, then SIGKILL 30 s later."""

import re
import signal
import subprocess
import sys
import time

import grpc
import pytest

from kubesplit import kubesplit_pb2 as pb
from kubesplit import kubesplit_pb2_grpc as pb_grpc
from kubesplit.runtime import exit_on_sigterm


def test_sigterm_raises_system_exit():
    previous = signal.getsignal(signal.SIGTERM)
    exit_on_sigterm()
    try:
        with pytest.raises(SystemExit) as e:
            signal.raise_signal(signal.SIGTERM)
        assert e.value.code == 143
    finally:
        signal.signal(signal.SIGTERM, previous)


@pytest.mark.skipif(sys.platform == "win32", reason="Windows can't deliver SIGTERM to a process")
def test_server_exits_promptly_on_sigterm_with_a_long_poll_open(tmp_path):
    args = "server --dataset fake --clients 1 --port 0 --metrics-port 0 --results-dir".split()
    proc = subprocess.Popen(
        [sys.executable, "-m", "kubesplit", *args, str(tmp_path)],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        for line in proc.stderr:
            if m := re.search(r"listening on :(\d+)", line):
                port = int(m.group(1))
                break
        else:
            pytest.fail(f"server exited early with code {proc.wait()}")

        # A client waiting for round 2 holds a server thread in the 20 s long-poll.
        with grpc.insecure_channel(f"localhost:{port}") as channel:
            poll = pb_grpc.SplitLearningStub(channel).WaitRound.future(
                pb.RoundRequest(client_id="t", last_round=1), timeout=30
            )
            time.sleep(1)  # let the call reach the server
            t0 = time.monotonic()
            proc.send_signal(signal.SIGTERM)
            _, err = proc.communicate(timeout=15)
            elapsed = time.monotonic() - t0
            poll.cancel()
    finally:
        proc.kill()

    # 143 means our handler ran; a process killed by the signal itself reports -15.
    assert proc.returncode == 143, err
    assert "SIGTERM received" in err
    assert elapsed < 5, f"took {elapsed:.1f}s; the long-poll held up exit"
