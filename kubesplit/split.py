"""Client and server halves of split training, independent of transport.

Both halves exchange only plain tensors, so the in-process baseline and the
gRPC services (Day 2) drive the exact same code:

    vanilla:   h = client.forward(x)        -> server.step(h, y, want_grad=True)
               client.backward(grad)        <- grad at the cut
    decoupled: h = client.local_step(x, y)  -> server.step(h, y, want_grad=False)
               (no message back; the client already updated from its aux loss)
"""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from kubesplit.model import AuxHead, ClientBottom, ServerTop

Mode = Literal["vanilla", "decoupled"]


def make_optimizer(params, lr: float) -> torch.optim.Optimizer:
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)


def client_modules(mode: Mode) -> nn.ModuleDict:
    """Everything a client owns and FedAvg averages: bottom, plus aux if decoupled."""
    if mode not in ("vanilla", "decoupled"):
        raise ValueError(f"unknown mode {mode!r}")
    mods = {"bottom": ClientBottom()}
    if mode == "decoupled":
        mods["aux"] = AuxHead()
    return nn.ModuleDict(mods)


class ClientWorker:
    def __init__(self, mode: Mode, lr: float = 0.05, device: str = "cpu"):
        self.mode = mode
        self.device = device
        self._mods = client_modules(mode).to(device)
        self.bottom = self._mods["bottom"]
        self.aux = self._mods["aux"] if "aux" in self._mods else None
        self.opt = make_optimizer(self._mods.parameters(), lr)
        self._pending: torch.Tensor | None = None

    def modules(self) -> nn.ModuleDict:
        return self._mods

    def train(self):
        self.modules().train()

    # -- vanilla -----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the bottom and hold the graph until backward(grad) arrives."""
        assert self.mode == "vanilla"
        self.opt.zero_grad(set_to_none=True)
        h = self.bottom(x.to(self.device))
        self._pending = h
        return h.detach()

    def backward(self, grad: torch.Tensor) -> None:
        assert self._pending is not None, "backward() without forward()"
        self._pending.backward(grad.to(self.device))
        self.opt.step()
        self._pending = None

    # -- decoupled ---------------------------------------------------------
    def local_step(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, float]:
        """One greedy update from the aux loss. Returns (activation, aux loss)."""
        assert self.mode == "decoupled"
        self.opt.zero_grad(set_to_none=True)
        h = self.bottom(x.to(self.device))
        loss = nn.functional.cross_entropy(self.aux(h), y.to(self.device))
        loss.backward()
        self.opt.step()
        return h.detach(), loss.item()


@dataclass
class ServerStepResult:
    loss: float
    correct: int
    grad: torch.Tensor | None  # gradient at the cut, vanilla only


class ServerWorker:
    def __init__(self, lr: float = 0.05, device: str = "cpu"):
        self.device = device
        self.top = ServerTop().to(device)
        self.opt = make_optimizer(self.top.parameters(), lr)

    def step(self, h: torch.Tensor, y: torch.Tensor, want_grad: bool) -> ServerStepResult:
        self.top.train()
        h = h.to(self.device).requires_grad_(want_grad)
        y = y.to(self.device)
        self.opt.zero_grad(set_to_none=True)
        logits = self.top(h)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        self.opt.step()
        return ServerStepResult(
            loss=loss.item(),
            correct=(logits.argmax(1) == y).sum().item(),
            grad=h.grad.detach() if want_grad else None,
        )


def fedavg(states: list[dict[str, torch.Tensor]], weights: list[float]) -> dict[str, torch.Tensor]:
    """Weighted average of state dicts (weights = client sample counts).

    Integer buffers such as BatchNorm's num_batches_tracked are not
    averageable, so they are taken from the first state.
    """
    if not states or len(states) != len(weights):
        raise ValueError("need one weight per state")
    total = float(sum(weights))
    out = {}
    for k, v0 in states[0].items():
        if v0.is_floating_point():
            out[k] = sum(s[k] * (w / total) for s, w in zip(states, weights, strict=True))
        else:
            out[k] = v0.clone()
    return out


@torch.no_grad()
def evaluate(bottom: nn.Module, top: nn.Module, loader, device: str = "cpu") -> float:
    """Test accuracy (%) of the global bottom + server top."""
    bottom.eval()
    top.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (top(bottom(x)).argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)
