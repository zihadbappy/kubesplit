"""Synchronous FedAvg rounds with a quorum and a timeout. No transport here.

A round closes as soon as `quorum` distinct clients have submitted, or when
`round_timeout` seconds pass with at least one submission. Late submissions
for a closed round are rejected; the straggler just picks up the next round.
A round that times out with no submissions restarts its timer.
"""

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from kubesplit.split import fedavg

log = logging.getLogger(__name__)

State = dict[str, torch.Tensor]


@dataclass
class RoundClosed:
    round: int
    reason: str  # "quorum" | "timeout"
    clients: list[str]
    num_samples: int
    seconds: float


class RoundCoordinator:
    def __init__(
        self,
        initial_state: State,
        expected_clients: int,
        total_rounds: int,
        quorum_pct: float = 100.0,
        round_timeout: float = 600.0,
        on_close: Callable[[RoundClosed, State], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not 0 < quorum_pct <= 100:
            raise ValueError("quorum_pct must be in (0, 100]")
        self.quorum = max(1, math.ceil(expected_clients * quorum_pct / 100))
        self.total_rounds = total_rounds
        self.round_timeout = round_timeout
        self.on_close = on_close
        self._clock = clock
        self._cv = threading.Condition()
        self._updates: dict[str, tuple[State, int]] = {}
        self.global_state = initial_state
        self.round = 1
        self.done = False
        self._opened = clock()

    def submit(self, client_id: str, round: int, state: State, num_samples: int) -> bool:
        """Record a client's weights for `round`. False if that round is over."""
        with self._cv:
            if self.done or round != self.round:
                return False
            self._updates[client_id] = (state, num_samples)  # a resubmit replaces
            if len(self._updates) >= self.quorum:
                self._close("quorum")
            return True

    def check_timeout(self) -> None:
        """Call periodically; closes the round if its deadline has passed."""
        with self._cv:
            if self.done or self._clock() - self._opened < self.round_timeout:
                return
            if not self._updates:
                log.warning("round %d timed out with no updates; waiting again", self.round)
                self._opened = self._clock()
                return
            self._close("timeout")

    def wait(self, last_round: int, timeout: float) -> tuple[int, bool, State | None]:
        """Block until a round newer than `last_round` opens or training ends.

        Returns (round, done, global_state); round is 0 if the wait timed out.
        """
        with self._cv:
            self._cv.wait_for(lambda: self.done or self.round > last_round, timeout)
            if self.done:
                return self.round, True, None
            if self.round > last_round:
                return self.round, False, self.global_state
            return 0, False, None

    def _close(self, reason: str) -> None:
        clients = sorted(self._updates)
        states = [self._updates[c][0] for c in clients]
        weights = [self._updates[c][1] for c in clients]
        self.global_state = fedavg(states, weights)
        info = RoundClosed(
            round=self.round,
            reason=reason,
            clients=clients,
            num_samples=sum(weights),
            seconds=self._clock() - self._opened,
        )
        self._updates = {}
        if self.on_close is not None:
            self.on_close(info, self.global_state)
        if self.round >= self.total_rounds:
            self.done = True
        else:
            self.round += 1
            self._opened = self._clock()
        self._cv.notify_all()
