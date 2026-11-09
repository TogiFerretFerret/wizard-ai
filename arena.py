"""Shared match-running code for tournament.py, challenge.py and league.py."""
from __future__ import annotations

import importlib.util
import inspect
import os
import random
import statistics
from concurrent.futures import ProcessPoolExecutor

from bots import HeuristicBot, MCTSBot, RandomBot
from engine import Bot, WizardGame

BUILTIN = ("random", "heuristic", "mcts[:ITERS]")


def load_bot(spec: str) -> Bot:
    """random | heuristic | mcts[:N] | path/to/file.py | path/to/file.py:ClassName"""
    if spec == "random":
        return RandomBot()
    if spec == "heuristic":
        return HeuristicBot()
    if spec.startswith("mcts"):
        _, _, it = spec.partition(":")
        return MCTSBot(iterations=int(it or 200))
    path, cls = (spec.rsplit(":", 1) if spec.count(":") and not spec.endswith(".py") else (spec, None))
    if not path.endswith(".py") or not os.path.exists(path):
        raise SystemExit(f"Can't find bot {spec!r}. Use one of {BUILTIN} or a path like players/alice.py")
    name = os.path.splitext(os.path.basename(path))[0]
    m = importlib.util.spec_from_file_location(f"player_{name}", path)
    mod = importlib.util.module_from_spec(m)
    m.loader.exec_module(mod)
    if cls is None:
        found = [c for _, c in inspect.getmembers(mod, inspect.isclass)
                 if issubclass(c, Bot) and c is not Bot and c.__module__ == mod.__name__]
        if len(found) != 1:
            raise SystemExit(f"{path}: expected exactly one Bot subclass, found {[c.__name__ for c in found]}. "
                             f"Use {path}:ClassName")
        bot_cls = found[0]
    else:
        bot_cls = getattr(mod, cls)
    bot = bot_cls()
    if getattr(bot, "name", "bot") in ("bot", "my_bot"):
        bot.name = name
    return bot


def _play(job):
    specs, seating, seed, n_rounds = job
    bots = [load_bot(specs[i]) for i in seating]
    g = WizardGame(bots, seed=seed, n_rounds=n_rounds)
    scores = g.play()
    out = {}
    for seat, i in enumerate(seating):
        out[i] = (scores[seat], g.think_time[seat], g.decisions[seat])
    return [out[i] for i in range(len(specs))]


def run_match(specs: list[str], deals: int = 8, n_rounds: int | None = None, seed: int = 0,
              workers: int | None = None) -> list[dict]:
    """Play `deals` deal-sets; each set is replayed with every seat rotation (fair luck).

    Returns one stats dict per entrant: mean, se, win (fraction), margin (score minus
    table average), ms (avg ms per decision), scores.
    """
    n = len(specs)
    jobs = [(specs, list(range(n))[k:] + list(range(n))[:k], seed * 100003 + d, n_rounds)
            for d in range(deals) for k in range(n)]
    if workers == 1:
        results = [_play(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers or os.cpu_count()) as ex:
            results = list(ex.map(_play, jobs))
    stats = []
    for i in range(n):
        sc = [r[i][0] for r in results]
        wins = 0.0
        for r in results:
            best = max(x[0] for x in r)
            if r[i][0] == best:
                wins += 1 / sum(x[0] == best for x in r)
        margin = [r[i][0] - statistics.mean(x[0] for x in r) for r in results]
        t = sum(r[i][1] for r in results); d = sum(r[i][2] for r in results)
        stats.append({"spec": specs[i], "mean": statistics.mean(sc),
                      "se": statistics.stdev(sc) / len(sc) ** .5 if len(sc) > 1 else 0.0,
                      "win": wins / len(results), "margin": statistics.mean(margin),
                      "ms": 1000 * t / max(d, 1), "scores": sc, "games": len(results)})
    return stats


def print_table(stats, names=None):
    names = names or [s["spec"] for s in stats]
    print(f"{'bot':<26}{'avg score':>10}{'± se':>7}{'win %':>8}{'ms/move':>9}")
    for i in sorted(range(len(stats)), key=lambda i: -stats[i]["mean"]):
        s = stats[i]
        print(f"{names[i]:<26}{s['mean']:>10.1f}{s['se']:>7.1f}{100 * s['win']:>7.1f}%{s['ms']:>9.1f}")
