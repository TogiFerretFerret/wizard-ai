"""Climb the challenge ladder with your bot.

  python challenge.py players/alice.py

Each level: your bot sits at a table with the listed opponents and must have the
highest average score over a seat-rotated set of games. Stops at the first level you fail.
"""
import argparse
import time

from arena import load_bot, print_table, run_match

LEVELS = [
    ("🃏  Apprentice", ["random", "random", "random"], "beat three random bots"),
    ("📜  Journeyman", ["heuristic", "heuristic", "heuristic"], "beat three rule-based bots"),
    ("🔮  Adept",      ["mcts:50", "heuristic", "heuristic"], "beat a small MCTS bot and two heuristics"),
    ("🧙  Archmage",   ["mcts:200", "mcts:200", "mcts:200"], "beat three full MCTS bots"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bot")
    ap.add_argument("--deals", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    me = load_bot(a.bot).name
    for i, (title, opps, desc) in enumerate(LEVELS, 1):
        print(f"\nLevel {i}: {title} — {desc}")
        t0 = time.time()
        stats = run_match([a.bot] + opps, deals=a.deals, seed=a.seed + i)
        print_table(stats, [f"{me} (you)"] + opps)
        best_opp = max(s["mean"] for s in stats[1:])
        if stats[0]["mean"] > best_opp:
            print(f"✅ PASSED  ({stats[0]['mean']:.0f} vs best opponent {best_opp:.0f}, {time.time() - t0:.0f}s)")
        else:
            print(f"❌ not yet ({stats[0]['mean']:.0f} vs best opponent {best_opp:.0f}). "
                  f"You reached level {i - 1}/{len(LEVELS)}.")
            return
    print("\n🏆 You cleared every level. Time to beat your clubmates: python league.py")


if __name__ == "__main__":
    main()
