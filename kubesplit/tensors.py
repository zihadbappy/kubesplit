"""Tensor <-> protobuf without pickle.

A tensor travels as (dtype name, shape, raw little-endian bytes). Only the
dtypes below are accepted, so a malicious peer can't make us build object
arrays or run any deserialization code.
"""

from collections import OrderedDict

import numpy as np
import torch

from kubesplit import kubesplit_pb2 as pb

_TO_NAME = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.float64: "float64",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.uint8: "uint8",
    torch.bool: "bool",
}
_NP_DTYPE = {name: np.dtype(name).newbyteorder("<") for name in _TO_NAME.values()}


def to_proto(t: torch.Tensor) -> pb.Tensor:
    try:
        name = _TO_NAME[t.dtype]
    except KeyError:
        raise ValueError(f"unsupported dtype {t.dtype}") from None
    arr = t.detach().cpu().contiguous().numpy().astype(_NP_DTYPE[name], copy=False)
    return pb.Tensor(dtype=name, shape=list(t.shape), data=arr.tobytes())


def from_proto(m: pb.Tensor) -> torch.Tensor:
    dtype = _NP_DTYPE.get(m.dtype)
    if dtype is None:
        raise ValueError(f"unsupported dtype {m.dtype!r}")
    shape = tuple(m.shape)
    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(m.data) != expected:
        raise ValueError(f"{len(m.data)} bytes does not match {m.dtype}{list(shape)}")
    arr = np.frombuffer(m.data, dtype=dtype).reshape(shape)
    # Copy: frombuffer views immutable bytes, and torch wants native byte order.
    return torch.from_numpy(arr.astype(dtype.newbyteorder("="), copy=True))


def state_to_proto(state: dict[str, torch.Tensor]) -> pb.StateDict:
    return pb.StateDict(
        entries=[pb.NamedTensor(name=k, tensor=to_proto(v)) for k, v in state.items()]
    )


def state_from_proto(m: pb.StateDict) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((e.name, from_proto(e.tensor)) for e in m.entries)
