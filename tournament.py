"""Pit 3-6 bots against each other at one table.

Bot specs:  random | heuristic | mcts[:ITERS] | players/alice.py | file.py:ClassName

  python tournament.py players/alice.py heuristic heuristic mcts:100
  python tournament.py mcts:100 heuristic heuristic heuristic --watch      # see every move of one game
"""
import argparse
import time

from arena import load_bot, print_table, run_match
from engine import WizardGame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bots", nargs="+")
    ap.add_argument("--deals", type=int, default=8, help="deal sets (each played once per seat rotation)")
    ap.add_argument("--rounds", type=int, default=None, help="rounds per game (default 10)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--watch", action="store_true", help="print every move of a single game")
    a = ap.parse_args()
    if a.watch:
        bots = [load_bot(s) for s in a.bots]
        print("final:", WizardGame(bots, seed=a.seed, n_rounds=a.rounds, verbose=True).play())
        return
    t0 = time.time()
    stats = run_match(a.bots, a.deals, a.rounds, a.seed, a.workers)
    print(f"\n{stats[0]['games']} games in {time.time() - t0:.1f}s\n")
    print_table(stats)


if __name__ == "__main__":
    main()
