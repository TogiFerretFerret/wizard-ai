"""Meeting demo: one command, pick a number.

  python demo.py

Needs numpy for the trained net (models/best.npz). Everything else is plain Python.
"""
from __future__ import annotations

import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "best.npz")
PY = sys.executable

MENU = """
🧙  Wizard AI — club demo
  1) Play against the trained AI          (you + AI + MCTS + heuristic)
  2) Watch the AI play a short game       (every card, every trick)
  3) Peek inside the AI's head            (what it thinks of a random hand)
  4) Mini tournament                      (AI vs MCTS vs rule-based vs random)
  5) Pass-and-play for 4 humans
  q) quit
"""


def run(*args):
    subprocess.run([PY, *args], cwd=HERE)


def peek():
    sys.path.insert(0, HERE)
    from engine import DECK, RoundState, SUIT_SYMBOLS, card_str
    from netbot import NumpyNet, net_search
    net = NumpyNet(MODEL)
    rng = random.Random()
    r = rng.choice([3, 4, 5, 6])
    deck = list(DECK); rng.shuffle(deck)
    s = RoundState(4, r, 3, [deck[i * r:(i + 1) * r] for i in range(4)], deck[4 * r])
    obs = s.observe(0)
    trump = SUIT_SYMBOLS[s.trump] if s.trump is not None else "none"
    print(f"\nRound with {r} cards. Trump card {card_str(s.trump_card)} (trump: {trump}). You bid first.")
    print("Hand:", "  ".join(card_str(c) for c in sorted(obs.hand)))
    print("The AI imagines hundreds of possible deals for the hidden cards and searches each one...\n")
    root = net_search(obs, net, 800, rng=rng)
    total = sum(ch.N for ch in root.children.values())
    scale = 20 + 10 * r
    for a, ch in sorted(root.children.items(), key=lambda kv: -kv[1].N):
        pct = 100 * ch.N / total
        print(f"  bid {a}: {'█' * int(pct / 2):<50} {pct:4.1f}% of search · expects {scale * ch.W / ch.N:+5.0f} pts")
    best = max(root.children, key=lambda a: root.children[a].N)
    print(f"\nAI's bid: {best}")
    input("\nEnter to go back to the menu... ")


def main():
    if not os.path.exists(MODEL):
        sys.exit("models/best.npz is missing. Clone the repo again or run train.py.")
    try:
        import numpy  # noqa: F401
    except ImportError:
        sys.exit("The trained AI needs numpy:  pip install numpy")
    net = f"net:{MODEL}:200"
    while True:
        print(MENU)
        choice = input("pick> ").strip().lower()
        if choice == "1":
            run("play.py", "human", net, "mcts:200", "heuristic", "--rounds", "6")
        elif choice == "2":
            run("tournament.py", net, "mcts:200", "heuristic", "random", "--watch", "--rounds", "4",
                "--seed", str(random.randrange(10_000)))
        elif choice == "3":
            peek()
        elif choice == "4":
            print("Playing 32 games (seats rotate so nobody gets lucky)... about a minute.")
            run("tournament.py", net, "mcts:200", "heuristic", "random", "--deals", "8")
        elif choice == "5":
            run("play.py", "human", "human", "human", "human", "--rounds", "5")
        elif choice in ("q", "quit", "exit"):
            return


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nbye!")
