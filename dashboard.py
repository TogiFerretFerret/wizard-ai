"""Live terminal dashboard for train.py.

  python dashboard.py            # refreshes every 2s, q to quit
  python dashboard.py --once     # print one plain-text snapshot (no curses)

Reads models/log.tsv, models/train_output.txt and models/progress.json (all written by
train.py), plus /proc and nvidia-smi.
"""
from __future__ import annotations

import argparse
import curses
import json
import os
import re
import shutil
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "models", "log.tsv")
OUT = os.path.join(HERE, "models", "train_output.txt")
PROG = os.path.join(HERE, "models", "progress.json")
SPARK = "▁▂▃▄▅▆▇█"


# ----------------------------------------------------------------------------- data
def read_log():
    rows = []
    try:
        with open(LOG) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if p and p[0].isdigit():
                    rows.append({"iter": int(p[0]), "net": float(p[1]), "mcts": float(p[2]),
                                 "margin": float(p[3]), "samples": int(p[4]), "mix": float(p[5]),
                                 "vp": float(p[6]), "vv": float(p[7]), "secs": float(p[8])})
    except (OSError, ValueError, IndexError):
        pass
    return rows


def read_progress():
    try:
        with open(PROG) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_output():
    try:
        with open(OUT) as f:
            lines = [l.rstrip("\n") for l in f if l.strip()]
        return lines, os.path.getmtime(OUT)
    except OSError:
        return [], None


def train_process():
    """(pid, elapsed seconds, total iters) of the main train.py process, or None."""
    try:
        out = subprocess.run(["ps", "-ww", "-eo", "pid,etimes,args"], capture_output=True, text=True).stdout
    except OSError:
        return None
    for line in out.splitlines():
        if "train.py" in line and "forkserver" not in line and "multiprocessing" not in line \
                and "dashboard" not in line and "python" in line:
            pid, et, args = line.split(None, 2)
            m = re.search(r"--iters[ =](\d+)", args)
            return int(pid), int(et), int(m.group(1)) if m else 10, "--quick" in args
    return None


def worker_count():
    try:
        out = subprocess.run(["ps", "-ww", "-eo", "args"], capture_output=True, text=True).stdout
        return sum("multiprocessing" in l and "wizard-ai" in l and ("forkserver" in l or "spawn" in l)
                   for l in out.splitlines())
    except OSError:
        return 0


_prev_cpu = None


def cpu_percent():
    global _prev_cpu
    with open("/proc/stat") as f:
        v = list(map(int, f.readline().split()[1:]))
    idle, total = v[3] + v[4], sum(v)
    pct = 0.0
    if _prev_cpu:
        di, dt = idle - _prev_cpu[0], total - _prev_cpu[1]
        pct = 100 * (1 - di / dt) if dt else 0.0
    _prev_cpu = (idle, total)
    return pct


def mem():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            info[k] = int(v.split()[0])
    total = info["MemTotal"] / 2**20
    return total - info["MemAvailable"] / 2**20, total


def gpu():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,"
                              "temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=3).stdout.strip()
        u, mu, mt, t, p = [x.strip() for x in out.split(",")]
        return {"util": float(u), "mem": float(mu) / 1024, "memt": float(mt) / 1024,
                "temp": float(t), "power": float(p) if p not in ("[N/A]", "") else 0.0}
    except Exception:
        return None


def fmt_t(s):
    s = int(max(0, s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def phase_info(lines, rows, proc):
    phase, detail = "waiting for train.py", ""
    for l in lines:
        if l.startswith("=== bootstrap"):
            phase, detail = "BOOTSTRAP", "plain-ISMCTS self-play"
        elif l.startswith("loaded") and "bootstrap" in l:
            phase, detail = "BOOTSTRAP", "loaded cached data"
        elif l.strip().startswith("trained on bootstrap"):
            detail = "evaluating bootstrap net"
        elif l.startswith("=== iteration"):
            m = re.match(r"=== iteration (\d+)", l)
            phase, detail = f"ITERATION {m.group(1)}", "net-guided self-play"
        elif l.strip().startswith("+") and "samples" in l:
            detail = "evaluating vs MCTS-200"
        elif l.startswith("=== final"):
            phase, detail = "FINAL", "tuning playout blend"
        elif l.startswith("best.npz uses"):
            phase, detail = "DONE", l
    if proc is None and phase not in ("DONE",):
        detail = (detail + "  (train.py is not running)").strip()
    return phase, detail


def bar(v, lo, hi, width):
    if hi <= lo:
        return ""
    n = int(round((v - lo) / (hi - lo) * width))
    return "█" * max(0, min(width, n))


# ----------------------------------------------------------------------------- render
def build(width):
    """Return list of (text, style) lines; style in {'', 'title', 'good', 'bad', 'dim', 'accent'}."""
    rows = read_log()
    lines, mtime = read_output()
    proc = train_process()
    phase, detail = phase_info(lines, rows, proc)
    prog = read_progress()
    if prog:
        phase, detail = prog["phase"].upper(), ""
        if proc is None and prog["phase"] != "done":
            detail = "train.py is not running"
    out = []
    W = max(60, width - 2)

    out.append((" 🧙 Wizard AI training dashboard ".center(W, "─"), "title"))
    elapsed = proc[1] if proc else None
    total_iters = proc[2] if proc else (rows[-1]["iter"] + 1 if rows else 10)
    status = f" phase: {phase}"
    if detail:
        status += f" · {detail}"
    out.append((status, "accent"))
    since = time.time() - mtime if mtime else 0
    eta = ""
    if rows:
        avg = sum(r["secs"] for r in rows[-3:]) / len(rows[-3:])
        left = max(0, total_iters - len(rows))
        if phase.startswith("ITERATION") and left:
            eta = f" · ETA ≈ {fmt_t(left * avg - min(since, avg))} + final tuning"
    out.append((f" running {fmt_t(elapsed) if elapsed else '--'} · last output {fmt_t(since)} ago · "
                f"iterations done {len(rows)}/{total_iters}{eta}", ""))
    if prog and prog.get("total"):
        d, t = prog["done"], prog["total"]
        el = time.time() - prog["phase_start"]
        unit = "games" if "self-play" in prog["phase"] else "matches"
        rate = d / el * 60 if el > 0 and d else 0
        p_eta = f" · ETA {fmt_t((t - d) / d * el)}" if d else ""
        bw = W - 50
        fill = int(d / t * bw)
        out.append((f" {prog['phase']}: [{'█' * fill}{'░' * (bw - fill)}] {d}/{t} {unit} · "
                    f"{rate:.0f}/min{p_eta}", "good"))
    elif prog and prog["phase"] != "done":
        out.append((f" {prog['phase']} … ({fmt_t(time.time() - prog['phase_start'])})", "good"))
    if rows:
        done = len(rows) / total_iters
        bw = W - 30
        out.append((f" all iterations [" + "█" * int(done * bw) + "░" * (bw - int(done * bw)) +
                    f"] {len(rows)}/{total_iters}", "dim"))
    out.append(("", ""))

    # system
    c = cpu_percent()
    mu, mt = mem()
    g = gpu()
    cw = 30
    out.append((f" CPU {c:5.1f}% {bar(c, 0, 100, cw):<{cw}} workers {worker_count():>2}   "
                f"RAM {mu:4.1f}/{mt:.0f} GB", ""))
    if g:
        out.append((f" GPU {g['util']:5.1f}% {bar(g['util'], 0, 100, cw):<{cw}} VRAM {g['mem']:.1f}/{g['memt']:.0f} GB  "
                    f"{g['temp']:.0f}°C  {g['power']:.0f} W", ""))
    else:
        out.append((" GPU  n/a (nvidia-smi not found)", "dim"))
    out.append(("", ""))

    # eval table + chart
    boot = None
    for l in lines:
        m = re.search(r"eval: netbot ([\d.-]+) vs mcts:200 ([\d.-]+)", l)
        if m and boot is None:
            boot = (float(m.group(1)), float(m.group(2)))
    out.append((" NetBot-200 vs MCTS-200  (avg game score; margin > 0 means the net bot wins)", "title"))
    pts = ([("boot", boot[0], boot[1], boot[0] - boot[1], None, None, None)] if boot else []) + \
          [(str(r["iter"]), r["net"], r["mcts"], r["margin"], r["mix"], r["vv"], r["samples"]) for r in rows]
    if not pts:
        out.append(("   no evaluations yet (first one comes after the bootstrap phase)", "dim"))
    else:
        best = max(p[3] for p in pts)
        lo, hi = min(0.0, min(p[3] for p in pts)), max(0.0, max(p[3] for p in pts))
        cw = max(10, W - 62)
        zero = int(round((0 - lo) / (hi - lo) * cw)) if hi > lo else 0
        out.append((f"   {'iter':>4} {'net':>6} {'mcts':>6} {'margin':>7} {'blend':>5} {'val v':>7}  chart", "dim"))
        for name, nb, mc, mg, mix, vv, ns in pts[-14:]:
            pos = int(round((mg - lo) / (hi - lo) * cw)) if hi > lo else zero
            cells = [" "] * (cw + 1)
            cells[zero] = "│"
            a, b = sorted((zero, pos))
            for i in range(a, b + 1):
                if i != zero or a == b:
                    cells[i] = "█"
            star = " ★" if mg == best else ""
            mix_s = f"{mix:.2f}" if mix is not None else "  -  "
            vv_s = f"{vv:.4f}" if vv is not None else "   -   "
            style = "good" if mg >= 0 else "bad"
            out.append((f"   {name:>4} {nb:6.1f} {mc:6.1f} {mg:+7.1f} {mix_s:>5} {vv_s:>7}  {''.join(cells)}{star}", style))
        if rows:
            vvs = [r["vv"] for r in rows]
            lo_, hi_ = min(vvs), max(vvs)
            spark = "".join(SPARK[int((v - lo_) / (hi_ - lo_ + 1e-12) * 7)] for v in vvs)
            out.append((f"   value loss trend {spark}  ({vvs[0]:.4f} → {vvs[-1]:.4f}, lower is better)", "dim"))
        out.append((f"   best margin so far: {best:+.1f}  → models/best.npz", "accent"))
    out.append(("", ""))
    out.append((" latest output", "title"))
    for l in lines[-8:]:
        out.append(("   " + l[:W - 4], "dim"))
    out.append(("", ""))
    out.append((" q quit · r refresh", "dim"))
    return out


def run_curses(stdscr, interval):
    curses.curs_set(0)
    curses.use_default_colors()
    styles = {"": 0}
    for i, (name, col) in enumerate([("title", curses.COLOR_CYAN), ("good", curses.COLOR_GREEN),
                                     ("bad", curses.COLOR_RED), ("accent", curses.COLOR_YELLOW),
                                     ("dim", curses.COLOR_WHITE)], 1):
        curses.init_pair(i, col, -1)
        styles[name] = curses.color_pair(i) | (curses.A_BOLD if name in ("title", "accent") else 0)
    styles["dim"] |= curses.A_DIM
    stdscr.timeout(int(interval * 1000))
    cpu_percent()
    while True:
        h, w = stdscr.getmaxyx()
        stdscr.erase()
        for y, (text, st) in enumerate(build(w)[:h - 1]):
            try:
                stdscr.addnstr(y, 0, text, w - 1, styles.get(st, 0))
            except curses.error:
                pass
        stdscr.refresh()
        k = stdscr.getch()
        if k in (ord("q"), ord("Q"), 27):
            return


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    if a.once:
        cpu_percent(); time.sleep(0.5)
        for text, _ in build(shutil.get_terminal_size((110, 40)).columns):
            print(text)
        return
    try:
        curses.wrapper(run_curses, a.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
