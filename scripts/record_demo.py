"""Record the README demo: a real run on kind, replayed as a terminal GIF.

    pip install matplotlib   # brings Pillow and the DejaVu Sans Mono font
    python scripts/record_demo.py --context kind-kubesplit

Runs the demo commands for real (install a run, watch rounds close, kill the
server, watch it recover), records every output line with its timestamp as an
asciicast v2 file (docs/demo.cast, playable with asciinema), then draws it as
docs/demo.gif. Pauses longer than --idle seconds are shortened, as asciinema
players do with idle_time_limit, so a 2-minute session plays in under a minute.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
COLS, ROWS = 120, 26  # server log lines are ~119 characters
# ANSI codes in the cast: a blue prompt, grey comments. The GIF renderer reads them too.
PROMPT_SEQ, COMMENT_ON, RESET = "\x1b[34m$\x1b[0m ", "\x1b[90m", "\x1b[0m"
FONT_PX, CELL_W, CELL_H, PAD = 15, 9, 19, 14
# Dark surface and text from the charts' palette; prompt blue, comments muted.
BG, FG, PROMPT, COMMENT, OK = "#1a1a19", "#f0efec", "#3987e5", "#a8a79f", "#199e70"
TYPE_DELAY = 0.035  # seconds per typed character


class Recorder:
    def __init__(self, context: str):
        self.context = context
        self.events: list[tuple[float, str]] = []  # (seconds since start, text)
        self.t0 = time.monotonic()

    def emit(self, text: str) -> None:
        self.events.append((time.monotonic() - self.t0, text))

    def type(self, line: str, comment: bool = False) -> None:
        """A command (or comment) typed at the prompt, a character at a time."""
        self.emit(COMMENT_ON if comment else PROMPT_SEQ)
        for ch in line:
            self.emit(ch)
            time.sleep(TYPE_DELAY)
        if comment:
            self.emit(RESET)
        self.emit("\n")

    def run(self, shown: str, argv: list[str], keep=lambda line: True) -> None:
        self.type(shown)
        p = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)  # fmt: skip
        for line in p.stdout:
            line = line.rstrip()
            if line and keep(line):
                self.emit(line + "\n")
        p.wait()


def wait_for_rounds(context: str, n: int) -> None:
    for _ in range(600):
        logs = subprocess.run(["kubectl", "--context", context, "logs", "deploy/kubesplit-server"],
                              capture_output=True, text=True).stdout  # fmt: skip
        if logs.count("closed by") >= n:
            return
        time.sleep(1)
    sys.exit(f"fewer than {n} rounds closed after 10 min")


def record(context: str) -> list[tuple[float, str]]:
    kc = ["kubectl", "--context", context]
    helm = ["helm", "--kube-context", context]
    # Start clean, off camera.
    subprocess.run([*helm, "uninstall", "kubesplit", "--wait", "--ignore-not-found"], cwd=ROOT)
    pods = [*kc, "get", "pods", "-l", "app.kubernetes.io/instance=kubesplit"]
    while subprocess.run([*pods, "-o", "name"], capture_output=True, text=True).stdout.strip():
        time.sleep(2)

    r = Recorder(context)
    r.type("# KubeSplit on kind: 4 client pods + 1 server, decoupled split learning,", comment=True)
    r.type("# 50 ms of added client-server latency (tc netem in an init container)", comment=True)
    # Shown exactly as run (decoupled is the chart's default mode), minus kube-context/timeout.
    flags = ["--set", "run.rounds=300", "--set", "run.netDelayMs=50", "--wait"]
    install = ["install", "kubesplit", "charts/kubesplit", "-f", "deploy/kind/values.yaml", *flags]
    r.run(
        "helm " + " ".join(install),
        [*helm, *install, "--timeout", "10m"],
        keep=lambda line: line.startswith(("NAME:", "STATUS:", "REVISION:")),
    )
    r.run("kubectl get pods", pods)
    wait_for_rounds(context, 4)
    r.type(
        "# rounds close once every client has trained its shard and sent its weights", comment=True
    )
    r.run(
        "kubectl logs deploy/kubesplit-server | grep closed",
        [*kc, "logs", "deploy/kubesplit-server"],
        keep=lambda line: "closed by" in line,
    )
    r.type("# kill the server mid-round: clients rejoin the replacement in-process", comment=True)
    r.run(
        "python scripts/pod_kill.py server",
        [sys.executable, "-u", "scripts/pod_kill.py", "server", "--context", context,
         "--at-round", "6"],
    )  # fmt: skip
    r.run("kubectl get pods", pods)
    r.emit("")
    return r.events


def compress(events: list[tuple[float, str]], idle: float) -> list[tuple[float, str]]:
    """Cap pauses at `idle` seconds and play typing 3x faster; output keeps its timing."""
    out, shift, last = [], 0.0, 0.0
    for t, text in events:
        gap = t - last
        typed = len(text) == 1 and text != "\n"
        keep = gap / 3 if typed else min(gap, idle)
        shift += gap - keep
        out.append((t - shift, text))
        last = t
    return out


def write_cast(events, path: Path, idle: float) -> None:
    header = {"version": 2, "width": COLS, "height": ROWS, "idle_time_limit": idle,
              "title": "KubeSplit on kind", "env": {"TERM": "xterm-256color"}}  # fmt: skip
    lines = [json.dumps(header)]
    lines += [json.dumps([round(t, 3), "o", s.replace("\n", "\r\n")]) for t, s in events]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


class Screen:
    """A minimal terminal: wrapped lines, scrolls at the bottom, one colour per line."""

    def __init__(self):
        self.lines: list[list] = [["", FG]]

    def feed(self, text: str) -> None:
        if text == PROMPT_SEQ:
            self.lines[-1] = ["$ ", PROMPT]
            return
        if text in (COMMENT_ON, RESET):
            if text == COMMENT_ON:
                self.lines[-1] = ["", COMMENT]
            return
        for ch in text:
            line = self.lines[-1]
            if ch == "\n":
                self.lines.append(["", FG])
            elif len(line[0]) >= COLS:  # wrap: the rest keeps the line's colour
                self.lines.append([ch, FG if line[1] == PROMPT else line[1]])
            else:
                line[0] += ch
                if line[1] == FG and line[0] == "recovered":
                    line[1] = OK
        self.lines = self.lines[-ROWS:]

    def draw(self, font) -> Image.Image:
        img = Image.new("RGB", (COLS * CELL_W + 2 * PAD, ROWS * CELL_H + 2 * PAD), BG)
        d = ImageDraw.Draw(img)
        for row, (text, colour) in enumerate(self.lines):
            y = PAD + row * CELL_H
            if colour == PROMPT and text.startswith("$ "):
                d.text((PAD, y), "$", font=font, fill=PROMPT)
                d.text((PAD + 2 * CELL_W, y), text[2:], font=font, fill=FG)
            else:
                d.text((PAD, y), text, font=font, fill=colour)
        return img


def write_gif(events, path: Path, hold: float = 6.0) -> None:
    font_dir = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    font = ImageFont.truetype(str(font_dir / "DejaVuSansMono.ttf"), FONT_PX)
    # One palette for every frame (from a swatch of all the colours), so nothing flickers.
    swatch = Image.new("RGB", (64, 64), BG)
    for i, c in enumerate((FG, PROMPT, COMMENT, OK)):
        ImageDraw.Draw(swatch).text((2, 2 + 14 * i), "Mg$#", font=font, fill=c)
    palette = swatch.quantize(colors=32, dither=Image.Dither.NONE)
    screen, frames, durations = Screen(), [], []
    shown_at = events[0][0]  # when the frame being built goes on screen
    for i, (t, text) in enumerate(events):
        screen.feed(text)
        nxt = events[i + 1][0] if i + 1 < len(events) else t + hold
        # Browsers stretch GIF frames shorter than 20 ms, so group fast events.
        if nxt - shown_at < 0.02 and i + 1 < len(events):
            continue
        frames.append(screen.draw(font).quantize(palette=palette, dither=Image.Dither.NONE))
        durations.append(round((nxt - shown_at) * 1000))
        shown_at = nxt
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   optimize=True, disposal=1)  # fmt: skip


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--context", default="kind-kubesplit")
    p.add_argument("--idle", type=float, default=1.5, help="longest pause kept, in seconds")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "demo")
    p.add_argument("--from-cast", type=Path, help="redraw the GIF from an existing .cast")
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.from_cast:
        raw = [
            json.loads(line) for line in a.from_cast.read_text(encoding="utf-8").splitlines()[1:]
        ]
        events = [(t, s.replace("\r\n", "\n")) for t, _, s in raw]
    else:
        events = record(a.context)
        write_cast(events, a.out.with_suffix(".cast"), a.idle)
    write_gif(compress(events, a.idle), a.out.with_suffix(".gif"))
    size = os.path.getsize(a.out.with_suffix(".gif")) / 1024
    print(f"wrote {a.out.with_suffix('.gif')} ({size:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
