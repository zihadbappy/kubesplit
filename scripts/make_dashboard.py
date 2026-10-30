"""Build the Grafana dashboard the chart ships: charts/kubesplit/dashboards/kubesplit.json.

    python scripts/make_dashboard.py

Edit panels here and rerun, rather than editing the JSON by hand. Every query
filters on $namespace and $release, so one dashboard covers several releases.
"""

import json
import pathlib

DS = {"type": "prometheus", "uid": "${datasource}"}
S = 'namespace="$namespace", job="$release-server"'
C = 'namespace="$namespace", job="$release-client"'
POD = 'namespace="$namespace", pod=~"$release-(server-.+|client-[0-9]+)"'
CTR = POD + ', container!=""'
# Categorical slots 1-3, dark steps (validated). "Work" is always blue, "waiting" orange.
BLUE, ORANGE, AQUA = "#3987e5", "#d95926", "#199e70"
GRAY = "#8e8e8e"

OUT = pathlib.Path(__file__).parent.parent / "charts/kubesplit/dashboards/kubesplit.json"
panels = []


def nid():
    return len(panels) + 1


def target(expr, legend="", ref="A"):
    return {"refId": ref, "datasource": DS, "expr": expr, "legendFormat": legend or "__auto"}


def color(name, hex_):
    return {
        "matcher": {"id": "byName", "options": name},
        "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": hex_}}],
    }


def row(title, y):
    panels.append(
        {
            "type": "row",
            "id": nid(),
            "title": title,
            "collapsed": False,
            "panels": [],
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
        }
    )


def stat(
    title, desc, expr, pos, unit="none", decimals=None, mappings=None, legend="", text_mode="value"
):
    defaults = {
        "unit": unit,
        "color": {"mode": "fixed", "fixedColor": "text"},
        "mappings": mappings or [],
    }
    if decimals is not None:
        defaults["decimals"] = decimals
    panels.append(
        {
            "type": "stat",
            "id": nid(),
            "title": title,
            "description": desc,
            "datasource": DS,
            "gridPos": pos,
            "targets": [target(expr, legend)],
            "fieldConfig": {"defaults": defaults, "overrides": []},
            "options": {
                "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                "colorMode": "none",
                "graphMode": "none",
                "textMode": text_mode,
                "justifyMode": "auto",
                "orientation": "auto",
                "wideLayout": True,
            },
        }
    )


def series(
    title,
    desc,
    targets,
    pos,
    unit,
    stacked=False,
    step=False,
    by_name=False,
    overrides=(),
    soft_max=None,
    limit_line=None,
    legend=True,
    interval="5s",
):
    custom = {
        "drawStyle": "line",
        "lineWidth": 2,
        "showPoints": "never",
        "spanNulls": False,
        "lineInterpolation": "stepAfter" if step else "linear",
        "fillOpacity": 25 if stacked else 0,
        "gradientMode": "none",
        "stacking": {"mode": "normal" if stacked else "none", "group": "A"},
        "axisSoftMin": 0,
        "axisBorderShow": False,
    }
    defaults = {
        "unit": unit,
        "min": 0,
        "custom": custom,
        "color": {"mode": "palette-classic-by-name" if by_name else "palette-classic"},
    }
    if soft_max is not None:
        custom["axisSoftMax"] = soft_max
    if limit_line is not None:  # a dashed reference line, e.g. 100% of the pod's limit
        custom["thresholdsStyle"] = {"mode": "dashed"}
        defaults["thresholds"] = {
            "mode": "absolute",
            "steps": [
                {"color": "transparent", "value": None},
                {"color": GRAY, "value": limit_line},
            ],
        }
    panels.append(
        {
            "type": "timeseries",
            "id": nid(),
            "title": title,
            "description": desc,
            "datasource": DS,
            "gridPos": pos,
            "interval": interval,  # min step; $__rate_interval is 4x this
            "targets": targets,
            "fieldConfig": {"defaults": defaults, "overrides": list(overrides)},
            "options": {
                "legend": {"showLegend": legend, "displayMode": "list", "placement": "bottom"},
                "tooltip": {"mode": "multi", "sort": "desc"},
            },
        }
    )


def pos(x, y, w, h):
    return {"x": x, "y": y, "w": w, "h": h}


# -- Run ----------------------------------------------------------------------
row("Run", 0)
y = 1
stat("Round", "Round the server is training now.", f"max(kubesplit_round{{{S}}})", pos(0, y, 3, 4))
stat(
    "Status",
    "Done once every round has closed.",
    f"max(kubesplit_training_done{{{S}}})",
    pos(3, y, 3, 4),
    mappings=[
        {
            "type": "value",
            "options": {"0": {"text": "Training", "index": 0}, "1": {"text": "Done", "index": 1}},
        }
    ],
)
stat(
    "Clients joined",
    "Distinct clients that have joined this run.",
    f"max(kubesplit_clients_joined{{{S}}})",
    pos(6, y, 3, 4),
)
stat(
    "Test accuracy",
    "Global model on the test set after the last closed round.",
    f"max(kubesplit_test_accuracy_ratio{{{S}}})",
    pos(9, y, 3, 4),
    unit="percentunit",
    decimals=1,
)
stat(
    "Avg round time",
    "Mean time from a round opening to closing, over this run.",
    f"sum(kubesplit_round_duration_seconds_sum{{{S}}}) / "
    f"(sum(kubesplit_round_duration_seconds_count{{{S}}}) > 0)",
    pos(12, y, 3, 4),
    unit="s",
    decimals=1,
)
stat(
    "Timed-out rounds",
    "Rounds closed by the timeout instead of a full quorum.",
    f'sum(kubesplit_rounds_total{{{S}, reason="timeout"}})',
    pos(15, y, 3, 4),
)
stat(
    "Late updates",
    "Client updates discarded because their round had already closed.",
    f"sum(kubesplit_late_updates_total{{{S}}})",
    pos(18, y, 3, 4),
)
stat(
    "Client rejoins",
    "Joins beyond each client's first: a client pod restarted mid-run.",
    f"sum(kubesplit_client_joins_total{{{S}}}) - sum(kubesplit_clients_joined{{{S}}})",
    pos(21, y, 3, 4),
)

# -- Model --------------------------------------------------------------------
row("Model quality", 5)
y = 6
series(
    "Test accuracy",
    "Global model on the test set, updated when each round closes.",
    [target(f"max(kubesplit_test_accuracy_ratio{{{S}}})", "test accuracy")],
    pos(0, y, 12, 7),
    "percentunit",
    step=True,
    soft_max=1,
    legend=False,
)
series(
    "Training loss",
    "Server-side loss over the last closed round.",
    [target(f"max(kubesplit_train_loss{{{S}}})", "train loss")],
    pos(12, y, 12, 7),
    "none",
    step=True,
    legend=False,
)

# -- Clients ------------------------------------------------------------------
row("Clients: where the time goes", 13)
y = 14
series(
    "Client time per batch",
    "Average over all clients. compute = the client's own forward/backward; blocked = "
    "waiting on the network and the server. Vanilla blocks on every gradient; "
    "decoupled only waits when the server applies backpressure.",
    [
        target(
            f"sum by (phase) (rate(kubesplit_client_batch_seconds_sum{{{C}}}[$__rate_interval]))"
            f" / (sum by (phase) (rate(kubesplit_client_batch_seconds_count{{{C}}}"
            f"[$__rate_interval])) > 0)",
            "{{phase}}",
        )
    ],
    pos(0, y, 8, 8),
    "s",
    stacked=True,
    overrides=[color("compute", BLUE), color("blocked", ORANGE)],
)
series(
    "Client throughput",
    "Training samples per second, per client pod.",
    [
        target(
            f"sum by (pod) (rate(kubesplit_client_samples_total{{{C}}}[$__rate_interval]))",
            "{{pod}}",
        )
    ],
    pos(8, y, 8, 8),
    "suffix: samples/s",
    by_name=True,
)
panels.append(
    {
        "type": "bargauge",
        "id": nid(),
        "title": "Time waiting at the round barrier",
        "description": "Share of each client's run spent idle, waiting for slower clients so "
        "the round can close. High values mean stragglers or uneven shards.",
        "datasource": DS,
        "gridPos": pos(16, y, 8, 8),
        "targets": [
            target(
                f"sum by (pod) (kubesplit_client_idle_seconds_sum{{{C}}}) / "
                f"((sum by (pod) (kubesplit_client_idle_seconds_sum{{{C}}}) + "
                f"sum by (pod) (kubesplit_client_round_seconds_sum{{{C}}})) > 0)",
                "{{pod}}",
            )
        ],
        "fieldConfig": {
            "defaults": {
                "unit": "percentunit",
                "min": 0,
                "max": 1,
                "decimals": 0,
                "color": {"mode": "fixed", "fixedColor": BLUE},
            },
            "overrides": [],
        },
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "orientation": "horizontal",
            "displayMode": "basic",
            "valueMode": "text",
            "showUnfilled": True,
            "namePlacement": "left",
            "sizing": "auto",
        },
    }
)

# -- Network ------------------------------------------------------------------
row("Network", 22)
y = 23
series(
    "Traffic by message kind",
    "Serialized bytes per second through the server. activations go up, gradients come "
    "back down (vanilla only), weights are the per-round model sync.",
    [
        target(
            f"sum by (kind) (rate(kubesplit_transferred_bytes_total{{{S}}}[$__rate_interval]))",
            "{{kind}}",
        )
    ],
    pos(0, y, 16, 8),
    "binBps",
    stacked=True,
    overrides=[color("activations", BLUE), color("gradients", ORANGE), color("weights", AQUA)],
)
stat(
    "Total transferred",
    "Bytes per message kind since the server started.",
    f"sum by (kind) (kubesplit_transferred_bytes_total{{{S}}})",
    pos(16, y, 8, 8),
    unit="bytes",
    decimals=1,
    legend="{{kind}}",
    text_mode="value_and_name",
)

# -- Server -------------------------------------------------------------------
row("Server", 31)
y = 32
series(
    "Server time per batch",
    "compute = forward/backward through the shared top model; queued = waiting for "
    "another client's batch to release it. Queueing grows with clients.",
    [
        target(
            f"sum(rate(kubesplit_server_step_seconds_sum{{{S}}}[$__rate_interval])) / "
            f"(sum(rate(kubesplit_server_step_seconds_count{{{S}}}[$__rate_interval])) > 0)",
            "compute",
            "A",
        ),
        target(
            f"sum(rate(kubesplit_server_lock_wait_seconds_sum{{{S}}}[$__rate_interval])) / "
            f"(sum(rate(kubesplit_server_lock_wait_seconds_count{{{S}}}[$__rate_interval]))"
            f" > 0)",
            "queued",
            "B",
        ),
    ],
    pos(0, y, 12, 8),
    "s",
    stacked=True,
    overrides=[color("compute", BLUE), color("queued", ORANGE)],
)
series(
    "Server throughput",
    "Samples per second through the server's top model.",
    [target(f"sum(rate(kubesplit_server_samples_total{{{S}}}[$__rate_interval]))", "samples/s")],
    pos(12, y, 12, 8),
    "suffix: samples/s",
    legend=False,
)

# -- Pods ---------------------------------------------------------------------
# cAdvisor (kubelet) metrics are scraped every 30 s, so these panels need a
# 30 s step; with 5 s, $__rate_interval holds < 2 samples and pods go missing.
KUBELET_STEP = "30s"
row("Pods: CPU and memory against their limits", 40)
y = 41
series(
    "CPU used",
    "Share of each pod's CPU limit in use. The dashed line is the limit.",
    [
        target(
            f"sum by (pod) (rate(container_cpu_usage_seconds_total{{{CTR}}}"
            f"[$__rate_interval])) / sum by (pod) (kube_pod_container_resource_limits"
            f'{{{POD}, resource="cpu"}})',
            "{{pod}}",
        )
    ],
    pos(0, y, 8, 8),
    "percentunit",
    by_name=True,
    soft_max=1,
    limit_line=1,
    interval=KUBELET_STEP,
)
series(
    "CPU throttled",
    "Share of CFS periods in which the pod hit its CPU limit and was paused. Sustained "
    "throttling means the limit, not the code, sets the speed.",
    [
        target(
            f"sum by (pod) (rate(container_cpu_cfs_throttled_periods_total{{{CTR}}}"
            f"[$__rate_interval])) / (sum by (pod) (rate(container_cpu_cfs_periods_total"
            f"{{{CTR}}}[$__rate_interval])) > 0)",
            "{{pod}}",
        )
    ],
    pos(8, y, 8, 8),
    "percentunit",
    by_name=True,
    soft_max=1,
    interval=KUBELET_STEP,
)
series(
    "Memory used",
    "Working set as a share of each pod's memory limit. At 100% the pod is OOM-killed.",
    [
        target(
            f"sum by (pod) (container_memory_working_set_bytes{{{CTR}}}) / sum by (pod) "
            f'(kube_pod_container_resource_limits{{{POD}, resource="memory"}})',
            "{{pod}}",
        )
    ],
    pos(16, y, 8, 8),
    "percentunit",
    by_name=True,
    soft_max=1,
    limit_line=1,
    interval=KUBELET_STEP,
)


def var(name, label, query, regex=""):
    return {
        "name": name,
        "label": label,
        "type": "query",
        "datasource": DS,
        "query": {"query": query, "refId": f"{name}-query"},
        "definition": query,
        "regex": regex,
        "refresh": 2,
        "sort": 1,
        "hide": 0,
        "includeAll": False,
        "multi": False,
        "current": {},
    }


dashboard = {
    "uid": "kubesplit",
    "title": "KubeSplit",
    "description": "Split-learning run: progress, where client time goes, traffic, "
    "server contention and pod resources.",
    "tags": ["kubesplit"],
    "editable": True,
    "graphTooltip": 1,
    "refresh": "10s",
    "time": {"from": "now-30m", "to": "now"},
    "timezone": "browser",
    "schemaVersion": 39,
    "version": 1,
    "annotations": {"list": []},
    "templating": {
        "list": [
            {
                "name": "datasource",
                "label": "Data source",
                "type": "datasource",
                "query": "prometheus",
                "regex": "",
                "hide": 0,
                "current": {},
            },
            var("namespace", "Namespace", "label_values(kubesplit_round, namespace)"),
            var(
                "release",
                "Release",
                'label_values(kubesplit_round{namespace="$namespace"}, job)',
                "/^(.*)-server$/",
            ),
        ]
    },
    "panels": panels,
}

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(dashboard, indent=2) + "\n", newline="\n")
print(f"wrote {OUT} ({len(panels)} panels)")
