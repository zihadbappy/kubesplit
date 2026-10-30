import threading

import pytest
import torch

from kubesplit.rounds import RoundCoordinator


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def st(v: float):
    return {"w": torch.tensor([v])}


def make(expected=3, rounds=2, quorum_pct=100.0, timeout=10.0):
    clock, closed = FakeClock(), []
    c = RoundCoordinator(
        st(0.0),
        expected_clients=expected,
        total_rounds=rounds,
        quorum_pct=quorum_pct,
        round_timeout=timeout,
        on_close=lambda info, state: closed.append((info, state)),
        clock=clock,
    )
    return c, clock, closed


def test_quorum_math():
    assert make(expected=3, quorum_pct=60)[0].quorum == 2
    assert make(expected=4, quorum_pct=75)[0].quorum == 3
    assert make(expected=4, quorum_pct=1)[0].quorum == 1
    with pytest.raises(ValueError):
        make(quorum_pct=0)


def test_full_quorum_waits_for_everyone_then_averages_by_samples():
    c, _, closed = make(expected=3)
    assert c.submit("a", 1, st(0.0), 1)
    assert c.submit("b", 1, st(3.0), 1)
    assert c.round == 1 and not closed
    assert c.submit("c", 1, st(6.0), 4)
    assert c.round == 2
    info, state = closed[0]
    assert info.reason == "quorum" and info.clients == ["a", "b", "c"]
    assert info.num_samples == 6
    torch.testing.assert_close(state["w"], torch.tensor([(0 + 3 + 24) / 6]))


def test_partial_quorum_closes_early():
    c, _, closed = make(expected=4, quorum_pct=50)
    c.submit("a", 1, st(1.0), 1)
    c.submit("b", 1, st(1.0), 1)
    assert c.round == 2 and closed[0][0].clients == ["a", "b"]


def test_resubmission_does_not_count_twice():
    c, _, _ = make(expected=2)
    c.submit("a", 1, st(1.0), 1)
    c.submit("a", 1, st(2.0), 1)
    assert c.round == 1


def test_timeout_closes_with_what_it_has():
    c, clock, closed = make(expected=3, timeout=10)
    c.submit("a", 1, st(2.0), 1)
    clock.t = 9.9
    c.check_timeout()
    assert c.round == 1
    clock.t = 10.0
    c.check_timeout()
    assert c.round == 2
    info, state = closed[0]
    assert info.reason == "timeout" and info.clients == ["a"] and info.seconds == 10.0
    torch.testing.assert_close(state["w"], torch.tensor([2.0]))


def test_timeout_with_no_updates_restarts_the_timer():
    c, clock, closed = make(timeout=10)
    clock.t = 15
    c.check_timeout()
    assert c.round == 1 and not closed
    c.submit("a", 1, st(1.0), 1)
    clock.t = 24.9  # new deadline is 15 + 10
    c.check_timeout()
    assert c.round == 1
    clock.t = 25
    c.check_timeout()
    assert c.round == 2


def test_late_update_is_rejected():
    c, _, _ = make(expected=1)
    assert c.submit("a", 1, st(1.0), 1)
    assert not c.submit("b", 1, st(9.0), 1)  # round 1 already closed
    assert c.submit("b", 2, st(9.0), 1)


def test_wait_and_done():
    c, _, closed = make(expected=1, rounds=2)
    assert c.wait(0, timeout=0)[:2] == (1, False)
    assert c.wait(1, timeout=0) == (0, False, None)  # nothing newer yet
    c.submit("a", 1, st(5.0), 1)
    rnd, done, state = c.wait(1, timeout=0)
    assert (rnd, done) == (2, False)
    torch.testing.assert_close(state["w"], torch.tensor([5.0]))
    c.submit("a", 2, st(7.0), 1)
    assert c.done and c.wait(2, timeout=0)[:2] == (2, True)
    assert not c.submit("a", 2, st(1.0), 1)
    assert len(closed) == 2


def test_stop_releases_a_blocked_wait():
    c, _, _ = make()
    result = []
    t = threading.Thread(target=lambda: result.append(c.wait(1, timeout=30)))
    t.start()
    c.stop()
    t.join(timeout=5)
    assert not t.is_alive()
    assert result == [(0, False, None)]  # "poll again", not "training is done"
