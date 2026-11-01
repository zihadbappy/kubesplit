"""Charts and tables from a benchmark directory (scripts/benchmark.py output).

    pip install matplotlib
    python scripts/plot_results.py benchmarks/gke --img docs/img

Writes, for each chart, a light and a dark PNG (<name>-light.png, <name>-dark.png)
for the README's <picture> tags, and <bench>/results.md: Markdown tables with
the mean and range over repeats per (mode, delay), plus recovery times and
rounds lost from any pod-kill*.csv. Only clean runs count (no restarts, one join per client).
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots 1-2 and surfaces, light and dark steps of a palette that
# was checked for colour-blind separation. Decoupled is slot 1 (blue).
THEMES = {
    "light": {
        "surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
        "decoupled": "#2a78d6", "vanilla": "#eb6834", "compute": "#2a78d6", "blocked": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19", "text": "#f0efec", "muted": "#c3c2b7", "grid": "#34332f",
        "decoupled": "#3987e5", "vanilla": "#d95926", "compute": "#3987e5", "blocked": "#d95926",
    },
}  # fmt: skip
MODES = ["vanilla", "decoupled"]


def load(bench: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    rows = [r for r in csv.DictReader((bench / "summary.csv").open()) if r["clean"] == "True"]
    runs = {
        r["run_id"]: list(csv.DictReader((bench / "runs" / f"{r['run_id']}.csv").open()))
        for r in rows
    }
    return rows, runs


def by_config(rows: list[dict]) -> dict[tuple[str, int], list[dict]]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["mode"], int(r["net_delay_ms"]))].append(r)
    return dict(sorted(groups.items(), key=lambda kv: (kv[0][1], MODES.index(kv[0][0]))))


def stat(rs: list[dict], key: str) -> tuple[float, float, float]:
    xs = [float(r[key]) for r in rs]
    return statistics.mean(xs), min(xs), max(xs)


def style(ax, t: dict, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(t["surface"])
    ax.figure.set_facecolor(t["surface"])
    ax.set_title(title, color=t["text"], loc="left", fontsize=12, pad=30)  # room for the legend
    ax.set_xlabel(xlabel, color=t["muted"])
    ax.set_ylabel(ylabel, color=t["muted"])
    ax.tick_params(colors=t["muted"], length=0)
    ax.grid(color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])


def legend(ax, t: dict) -> None:
    """One row between the title and the plot, so it never covers data."""
    ax.legend(
        frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
        labelcolor=t["text"], borderaxespad=0.2,
    )  # fmt: skip


def round_time_vs_delay(groups, t: dict, ax) -> None:
    for mode in MODES:
        pts = [(d, *stat(rs, "steady_round_s")) for (m, d), rs in groups.items() if m == mode]
        if not pts:
            continue
        xs, means, lo, hi = zip(*pts, strict=True)
        err = [
            [m - lo_ for m, lo_ in zip(means, lo, strict=True)],
            [h - m for m, h in zip(means, hi, strict=True)],
        ]
        ax.errorbar(
            xs, means, yerr=err, color=t[mode], linewidth=2, marker="o", markersize=8,
            markeredgecolor=t["surface"], markeredgewidth=2, capsize=4, label=mode,
        )  # fmt: skip
        ax.annotate(
            f"{mode}  {means[-1]:.1f} s", (xs[-1], means[-1]), xytext=(8, 0),
            textcoords="offset points", va="center", color=t["text"],
        )  # fmt: skip
    style(
        ax,
        t,
        "Seconds per round as client-server latency grows",
        "added latency (ms)",
        "seconds per round",
    )
    ax.set_ylim(bottom=0)
    ax.set_xticks(sorted({d for _, d in groups}))
    legend(ax, t)


def batch_split(groups, t: dict, ax) -> None:
    labels, compute, blocked = [], [], []
    for (mode, delay), rs in groups.items():
        labels.append(f"{mode}, {delay} ms")
        compute.append(stat(rs, "client_compute_ms")[0])
        blocked.append(stat(rs, "client_blocked_ms")[0])
    ys = range(len(labels))
    bar = {"height": 0.6, "edgecolor": t["surface"], "linewidth": 2}  # 2px gap between segments
    ax.barh(ys, compute, color=t["compute"], label="compute (local)", **bar)
    ax.barh(
        ys, blocked, left=compute, color=t["blocked"], label="blocked (network + server)", **bar
    )
    for y, c, b in zip(ys, compute, blocked, strict=True):
        ax.annotate(
            f"{c + b:.0f} ms", (c + b, y), xytext=(6, 0), textcoords="offset points",
            va="center", color=t["text"],
        )  # fmt: skip
    ax.set_yticks(list(ys), labels)
    ax.invert_yaxis()
    style(ax, t, "Where a client's time per batch goes", "milliseconds per batch", "")
    ax.grid(axis="y", visible=False)
    legend(ax, t)


def accuracy_over_time(groups, runs, delay: int, t: dict, ax) -> None:
    for mode in MODES:
        rs = groups.get((mode, delay), [])
        if not rs:
            continue
        curves = [runs[r["run_id"]] for r in rs]
        n = min(len(c) for c in curves)
        xs = [
            statistics.mean(float(c[i]["elapsed_seconds"]) for c in curves) / 60 for i in range(n)
        ]
        ys = [statistics.mean(float(c[i]["test_acc"]) for c in curves) for i in range(n)]
        ax.plot(
            xs, ys, color=t[mode], linewidth=2, marker="o", markersize=8,
            markeredgecolor=t["surface"], markeredgewidth=2, label=mode,
        )  # fmt: skip
        ax.annotate(
            f"{mode}  {ys[-1]:.1f}%", (xs[-1], ys[-1]), xytext=(8, 0),
            textcoords="offset points", va="center", color=t["text"],
        )  # fmt: skip
    style(
        ax,
        t,
        f"Test accuracy against wall-clock time, {delay} ms latency",
        "minutes",
        "test accuracy (%)",
    )
    ax.set_xlim(left=0)
    legend(ax, t)


def save(fig, img: Path, name: str, theme: str) -> None:
    # bbox_inches="tight" keeps the end-of-line labels that sit past the axes.
    path = img / f"{name}-{theme}.png"
    fig.savefig(path, dpi=144, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def tables(groups) -> str:
    def cell(rs, key, fmt):
        mean, lo, hi = stat(rs, key)
        return f"{mean:{fmt}} ({lo:{fmt}}–{hi:{fmt}})" if len(rs) > 1 else f"{mean:{fmt}}"

    columns = [
        "mode", "latency", "runs", "s / round", "final test acc %",
        "compute ms / batch", "blocked ms / batch", "MiB up / down per run",
    ]  # fmt: skip
    out = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for (mode, delay), rs in groups.items():
        out.append(
            f"| {mode} | {delay} ms | {len(rs)} | {cell(rs, 'steady_round_s', '.1f')} | "
            f"{cell(rs, 'final_test_acc', '.1f')} | {cell(rs, 'client_compute_ms', '.0f')} | "
            f"{cell(rs, 'client_blocked_ms', '.0f')} | "
            f"{stat(rs, 'mib_up')[0]:.0f} / {stat(rs, 'mib_down')[0]:.0f} |"
        )
    return "\n".join(out)


def pod_kill_tables(bench: Path) -> str:
    """Seconds from the delete to each recovery stage, mean (range), per file and target kind."""
    stages = ["pod_running", "server_up", "rejoined", "round_closed"]
    out = []
    for path in sorted(bench.glob("pod-kill*.csv")):
        rows = list(csv.DictReader(path.open()))
        out += [f"`{path.name}`", "", "| killed | n | " + " | ".join(stages) + " | rounds lost |",
                "|" + "---|" * (len(stages) + 3)]  # fmt: skip
        for kind in ("client", "server"):
            rs = [r for r in rows if r["target"].startswith(kind)]
            if not rs:
                continue
            cells = []
            for s in stages:
                xs = [float(r[s]) for r in rs if r[s]]
                cells.append(
                    f"{statistics.mean(xs):.1f} ({min(xs):.1f}–{max(xs):.1f})" if xs else "–"
                )
            # Closed rounds the run lost: a server without a checkpoint starts over.
            lost = [int(r["rounds_lost"]) for r in rs]
            cells.append(f"{statistics.mean(lost):.1f} ({min(lost)}–{max(lost)})")
            out.append(f"| {kind} | {len(rs)} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("bench", type=Path, help="directory with summary.csv and runs/")
    p.add_argument("--img", type=Path, default=Path("docs/img"))
    p.add_argument("--headline-delay", type=int, default=50)
    a = p.parse_args()
    a.img.mkdir(parents=True, exist_ok=True)
    rows, runs = load(a.bench)
    groups = by_config(rows)

    for theme, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        round_time_vs_delay(groups, t, ax)
        save(fig, a.img, "round-time-vs-latency", theme)

        fig, ax = plt.subplots(figsize=(8, 0.6 * len(groups) + 1.6))
        batch_split(groups, t, ax)
        save(fig, a.img, "batch-time-split", theme)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        accuracy_over_time(groups, runs, a.headline_delay, t, ax)
        save(fig, a.img, "accuracy-over-time", theme)

    md = a.bench / "results.md"
    md.write_text(
        "<!-- Generated by scripts/plot_results.py -->\n\n## Training runs\n\n"
        + tables(groups)
        + "\n\n## Pod kills (seconds after the delete)\n\n"
        + pod_kill_tables(a.bench),
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {md} and {len(THEMES) * 3} charts in {a.img}")


if __name__ == "__main__":
    main()
