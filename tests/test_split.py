import torch

from kubesplit.model import CUT_SHAPE, AuxHead, ClientBottom, ServerTop
from kubesplit.split import ClientWorker, ServerWorker, fedavg


def batch(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, 32, 32, generator=g), torch.randint(0, 10, (n,), generator=g)


def params(m):
    return torch.cat([p.detach().flatten().clone() for p in m.parameters()])


def test_model_shapes():
    x, _ = batch()
    h = ClientBottom()(x)
    assert h.shape == (8, *CUT_SHAPE)
    assert ServerTop()(h).shape == (8, 10)
    assert AuxHead()(h).shape == (8, 10)


def test_vanilla_split_gradients_match_unsplit_network():
    torch.manual_seed(0)
    client, server = ClientWorker("vanilla", lr=0.1), ServerWorker(lr=0.1)
    x, y = batch()

    # Reference: the same weights trained as one unsplit network.
    ref_bottom = ClientBottom()
    ref_bottom.load_state_dict(client.bottom.state_dict())
    ref_top = ServerTop()
    ref_top.load_state_dict(server.top.state_dict())
    torch.nn.functional.cross_entropy(ref_top(ref_bottom(x)), y).backward()
    ref_grads = [p.grad.clone() for p in ref_bottom.parameters()]

    client.opt.step = lambda: None  # skip the update so we can compare raw grads
    h = client.forward(x)
    res = server.step(h, y, want_grad=True)
    assert res.grad.shape == h.shape
    client.backward(res.grad)
    for p, g in zip(client.bottom.parameters(), ref_grads, strict=True):
        torch.testing.assert_close(p.grad, g, rtol=1e-4, atol=1e-6)


def test_decoupled_client_updates_without_server():
    client = ClientWorker("decoupled", lr=0.1)
    before = params(client.bottom)
    x, y = batch()
    h, loss = client.local_step(x, y)
    assert not torch.equal(before, params(client.bottom))
    assert not h.requires_grad and h.grad_fn is None  # nothing flows back
    assert loss > 0


def test_decoupled_server_returns_no_grad():
    server = ServerWorker(lr=0.1)
    before = params(server.top)
    res = server.step(torch.randn(8, *CUT_SHAPE), torch.randint(0, 10, (8,)), want_grad=False)
    assert res.grad is None
    assert not torch.equal(before, params(server.top))


def test_only_decoupled_client_has_aux_head():
    assert ClientWorker("vanilla").aux is None
    assert set(ClientWorker("decoupled").modules().state_dict()) > set(
        ClientWorker("vanilla").modules().state_dict()
    )


def test_fedavg_weighted_mean():
    a = {"w": torch.tensor([0.0, 0.0]), "n": torch.tensor(5)}
    b = {"w": torch.tensor([4.0, 8.0]), "n": torch.tensor(9)}
    out = fedavg([a, b], [3, 1])
    torch.testing.assert_close(out["w"], torch.tensor([1.0, 2.0]))
    assert out["n"].item() == 5  # integer buffers are copied, not averaged


def test_fedavg_of_identical_models_is_identity():
    sd = ClientWorker("decoupled").modules().state_dict()
    out = fedavg([sd, sd, sd], [1, 2, 3])
    for k in sd:
        torch.testing.assert_close(out[k], sd[k])
