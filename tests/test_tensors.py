import pytest
import torch

from kubesplit import kubesplit_pb2 as pb
from kubesplit.split import ClientWorker
from kubesplit.tensors import from_proto, state_from_proto, state_to_proto, to_proto


@pytest.mark.parametrize(
    "t",
    [
        torch.randn(4, 64, 8, 8),
        torch.randn(3, 5).half(),
        torch.randn(2).double(),
        torch.randint(0, 10, (64,)),
        torch.randint(0, 10, (7,), dtype=torch.int32),
        torch.randint(0, 255, (2, 3), dtype=torch.uint8),
        torch.tensor([True, False]),
        torch.tensor(3.5),  # 0-d
        torch.empty(0, 4),
    ],
)
def test_roundtrip(t):
    m = to_proto(t)
    assert len(m.data) == t.numel() * t.element_size()  # raw bytes, no framing
    out = from_proto(pb.Tensor.FromString(m.SerializeToString()))
    assert out.dtype == t.dtype and out.shape == t.shape
    assert torch.equal(out, t)


def test_non_contiguous_input():
    t = torch.arange(12.0).reshape(3, 4).T
    assert torch.equal(from_proto(to_proto(t)), t)


def test_result_is_writable_and_independent():
    m = to_proto(torch.zeros(3))
    out = from_proto(m)
    out += 1  # would fail on a read-only view of the message bytes
    assert torch.equal(from_proto(m), torch.zeros(3))


def test_rejects_unknown_dtype():
    with pytest.raises(ValueError):
        from_proto(pb.Tensor(dtype="object", shape=[1], data=b"\x00" * 8))
    with pytest.raises(ValueError):
        to_proto(torch.zeros(2, dtype=torch.complex64))


def test_rejects_size_mismatch():
    with pytest.raises(ValueError):
        from_proto(pb.Tensor(dtype="float32", shape=[2, 2], data=b"\x00" * 12))


def test_state_dict_roundtrip_keeps_order_and_int_buffers():
    sd = ClientWorker("decoupled").modules().state_dict()
    out = state_from_proto(state_to_proto(sd))
    assert list(out) == list(sd)
    for k in sd:
        assert out[k].dtype == sd[k].dtype
        assert torch.equal(out[k], sd[k])
