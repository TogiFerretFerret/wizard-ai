"""Pit bots against each other.

Bot specs:
  random | heuristic | mcts[:ITERATIONS]          built-in bots
  path/to/file.py:ClassName                        your own Bot subclass

Examples:
  python tournament.py random heuristic mcts:200 heuristic --games 20
  python tournament.py alice.py:AliceBot bob.py:BobBot carol.py:CarolBot dan.py:DanBot
  python tournament.py mcts:100 heuristic heuristic heuristic --games 1 --verbose
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import os
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

from bots import HeuristicBot, MCTSBot, RandomBot
from engine import WizardGame


def load_bot(spec: str):
    if spec == "random":
        return RandomBot()
    if spec == "heuristic":
        return HeuristicBot()
    if spec.startswith("mcts"):
        _, _, it = spec.partition(":")
        return MCTSBot(iterations=int(it or 300))
    path, _, cls = spec.rpartition(":")
    if not path.endswith(".py"):
        raise SystemExit(f"unknown bot spec {spec!r}")
    mod_name = os.path.splitext(os.path.basename(path))[0]
    m = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(m)
    m.loader.exec_module(mod)
    bot = getattr(mod, cls)()
    if getattr(bot, "name", "bot") == "bot":
        bot.name = cls
    return bot


def run_one(args):
    specs, seating, seed, n_rounds, verbose = args
    bots = [load_bot(specs[i]) for i in seating]
    g = WizardGame(bots, seed=seed, n_rounds=n_rounds, verbose=verbose)
    scores = g.play()
    by_entrant = [0] * len(specs)
    for seat, i in enumerate(seating):
        by_entrant[i] = scores[seat]
    return by_entrant


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bots", nargs="+", help="3-6 bot specs")
    ap.add_argument("--games", type=int, default=8, help="games per seating rotation block")
    ap.add_argument("--rounds", type=int, default=None, help="rounds per game (default: full game)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--verbose", action="store_true", help="print every move (use with --games 1)")
    a = ap.parse_args()
    n = len(a.bots)
    # rotate seats so nobody benefits from a lucky seat; same deal seed across a rotation
    rotations = [list(range(n))[k:] + list(range(n))[:k] for k in range(n)]
    jobs = []
    for g in range(a.games):
        for rot in rotations:
            jobs.append((a.bots, rot, a.seed * 100003 + g, a.rounds, a.verbose))
        if a.verbose:
            jobs = jobs[:1]
            break
    names = [load_bot(s).name for s in a.bots]
    t0 = time.time()
    if a.verbose or a.workers == 1:
        results = [run_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(a.workers) as ex:
            results = list(ex.map(run_one, jobs))
    wins = [0.0] * n
    for r in results:
        best = max(r)
        winners = [i for i in range(n) if r[i] == best]
        for i in winners:
            wins[i] += 1 / len(winners)
    print(f"\n{len(results)} games in {time.time() - t0:.1f}s\n")
    print(f"{'entrant':<28}{'avg score':>10}{'± se':>8}{'win %':>8}")
    order = sorted(range(n), key=lambda i: -statistics.mean(r[i] for r in results))
    for i in order:
        col = [r[i] for r in results]
        se = statistics.stdev(col) / len(col) ** 0.5 if len(col) > 1 else 0
        print(f"{names[i] + ' (' + a.bots[i] + ')':<28}{statistics.mean(col):>10.1f}{se:>8.1f}"
              f"{100 * wins[i] / len(results):>7.1f}%")


if __name__ == "__main__":
    main()
