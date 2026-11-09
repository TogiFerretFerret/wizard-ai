"""Club league: every bot in players/ (plus baselines) plays many random 4-player tables.

  python league.py                    # writes LEADERBOARD.md
  python league.py --tables 40 --no-baselines

Ranking uses *margin*: your score minus the table average, averaged over every game
you played. That keeps it fair even though each table has different opponents.
"""
import argparse
import datetime
import glob
import itertools
import random
import statistics
import time

from arena import load_bot, run_match

BASELINES = ["heuristic", "mcts:200"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="players")
    ap.add_argument("--tables", type=int, default=20, help="random 4-player tables to play")
    ap.add_argument("--deals", type=int, default=2, help="deal sets per table (x4 seat rotations)")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="LEADERBOARD.md")
    a = ap.parse_args()
    entrants = sorted(glob.glob(f"{a.dir}/*.py"))
    entrants = [e for e in entrants if not e.split("/")[-1].startswith("_")]
    if not a.no_baselines:
        entrants += BASELINES
    if len(entrants) < 4:
        entrants += ["heuristic"] * (4 - len(entrants))
    names = {}
    for e in entrants:
        names[e] = load_bot(e).name
        print(f"  entrant: {names[e]:<20} ({e})")
    rng = random.Random(a.seed)
    combos = list(itertools.combinations(range(len(entrants)), 4))
    rng.shuffle(combos)
    # cycle through all distinct tables before repeating
    tables = [combos[i % len(combos)] for i in range(a.tables)]
    agg = {i: {"margin": [], "scores": [], "wins": 0.0, "games": 0, "ms": []} for i in range(len(entrants))}
    t0 = time.time()
    for k, tbl in enumerate(tables, 1):
        specs = [entrants[i] for i in tbl]
        stats = run_match(specs, deals=a.deals, seed=a.seed * 1000 + k)
        for i, s in zip(tbl, stats):
            agg[i]["margin"].append(s["margin"]); agg[i]["scores"] += s["scores"]
            agg[i]["wins"] += s["win"] * s["games"]; agg[i]["games"] += s["games"]; agg[i]["ms"].append(s["ms"])
        print(f"  table {k}/{len(tables)}: " + ", ".join(f"{names[e]} {s['mean']:.0f}" for e, s in zip(specs, stats)))
    rows = []
    for i, d in agg.items():
        if d["games"]:
            rows.append((statistics.mean(d["margin"]), names[entrants[i]], statistics.mean(d["scores"]),
                         100 * d["wins"] / d["games"], d["games"], statistics.mean(d["ms"])))
    rows.sort(reverse=True)
    lines = [f"# 🧙 Wizard AI League", "",
             f"_Updated {datetime.datetime.now():%Y-%m-%d %H:%M} · {len(tables)} tables · {time.time() - t0:.0f}s_", "",
             "| # | bot | margin vs table | avg score | win % | games | ms/move |",
             "|---|---|---:|---:|---:|---:|---:|"]
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    for r, (m, nm, sc, w, g, ms) in enumerate(rows, 1):
        lines.append(f"| {medals.get(r, r)} | {nm} | {m:+.1f} | {sc:.1f} | {w:.0f}% | {g} | {ms:.1f} |")
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
