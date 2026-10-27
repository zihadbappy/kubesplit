"""The /metrics endpoint serves the Prometheus text format Prometheus will scrape."""

from urllib.request import urlopen

import pytest

from kubesplit.metrics import ClientMetrics, ServerMetrics


def scrape(metrics) -> str:
    port = metrics.serve(0, "127.0.0.1")  # port 0 = any free port
    with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=10) as r:
        assert r.status == 200
        return r.read().decode()


def test_server_endpoint_lists_every_series_from_the_start():
    text = scrape(ServerMetrics())
    for line in (
        'kubesplit_transferred_bytes_total{kind="activations"} 0.0',
        'kubesplit_transferred_bytes_total{kind="gradients"} 0.0',
        'kubesplit_transferred_bytes_total{kind="weights"} 0.0',
        'kubesplit_rounds_total{reason="quorum"} 0.0',
        'kubesplit_rounds_total{reason="timeout"} 0.0',
        "# TYPE kubesplit_round_duration_seconds histogram",
        "# TYPE kubesplit_round gauge",
    ):
        assert line in text


def test_client_endpoint_has_only_client_metrics():
    text = scrape(ClientMetrics())
    assert 'kubesplit_client_batch_seconds_count{phase="blocked"} 0.0' in text
    assert "kubesplit_transferred_bytes" not in text  # server-only


@pytest.mark.parametrize(
    ("cls", "attr", "name"),
    [
        (ServerMetrics, "server_samples", "kubesplit_server_samples_total"),
        (ClientMetrics, "samples", "kubesplit_client_samples_total"),
    ],
)
def test_instances_do_not_share_state(cls, attr, name):
    a, b = cls(), cls()
    getattr(a, attr).inc(5)
    assert a.registry.get_sample_value(name) == 5
    assert b.registry.get_sample_value(name) == 0
