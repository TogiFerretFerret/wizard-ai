"""Play Wizard in the terminal against bots (or pass-and-play with friends).

  python play.py                                  # you vs mcts, heuristic, heuristic
  python play.py human mcts:400 mcts:400 mcts:400 # you vs three strong bots
  python play.py human human heuristic players/alice.py --names Ana Ben   # hot-seat
  python play.py --rounds 3                       # quick game

During your turn: type the number of a card or bid,  ?  for an AI hint,  r  for rules.
"""
from __future__ import annotations

import argparse
import os
import sys

from arena import load_bot
from bots import ismcts
from engine import (SUIT_SYMBOLS, Bot, Observation, WizardGame, card_str, is_jester,
                    is_wizard, rank, suit, trick_winner)

COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
RULES = """
  Wizards (Aces) always win - the FIRST Wizard played takes the trick.
  Jesters (2s) always lose. If every card is a Jester, the LAST Jester wins.
  You must follow the led suit if you can. Wizards and Jesters can always be played.
  Otherwise: highest trump wins, else highest card of the led suit.
  Bidding: total bids can't equal the cards dealt - the dealer is forced to break it.
  Score: exact bid = 20 + 10 per trick.  Miss = -10 per trick off.
"""


def c(text, code):
    return f"\033[{code}m{text}\033[0m" if COLOR else text


def pretty(card):
    s = card_str(card)
    if is_wizard(card):
        return c(s, "1;35")
    if is_jester(card):
        return c(s, "2")
    return c(s, "31") if suit(card) in (1, 2) else s


def hand_order(card):
    if is_wizard(card):
        return (0, 0)
    if is_jester(card):
        return (9, 0)
    return (1 + suit(card), -rank(card))


class HumanBot(Bot):
    def __init__(self, name, names_ref, hot_seat):
        self.name = name
        self.names = names_ref       # list of everyone's names (filled in by main)
        self.hot_seat = hot_seat
        self.seen_tricks = 0
        self.seen_round = 0

    def label(self, p):
        return "You" if p == self.seat and not self.hot_seat else self.names[p]

    def new_game(self, seat, n_players):
        self.seat = seat

    def act(self, obs: Observation):
        if obs.round_index != self.seen_round:
            self.seen_round, self.seen_tricks = obs.round_index, 0
        if self.hot_seat:
            print("\033[2J\033[H" if COLOR else "\n" * 40, end="")
            input(f"Pass to {c(self.name, '1')} and press Enter... ")
        self.show(obs)
        while True:
            prompt = "Your bid" if obs.phase == "bid" else "Play card #"
            raw = input(f"{c(prompt, '1')} ({'?' } hint, r rules): ").strip().lower()
            if raw == "r":
                print(RULES)
                continue
            if raw == "?":
                self.hint(obs)
                continue
            try:
                k = int(raw)
            except ValueError:
                print("  type a number")
                continue
            if obs.phase == "bid":
                if k in obs.legal_actions:
                    return k
                print(f"  can't bid {k}. allowed: {obs.legal_actions}")
            else:
                if not 1 <= k <= len(self.shown_hand):
                    print(f"  pick a card number from 1 to {len(self.shown_hand)}")
                elif self.shown_hand[k - 1] in obs.legal_actions:
                    return self.shown_hand[k - 1]
                else:
                    print("  you can't play that one: you must follow suit if you can")

    def show(self, obs):
        n = obs.n_players
        # tricks finished since my last turn
        for t, w in obs.history[self.seen_tricks:]:
            plays = "  ".join(f"{self.label(p)}: {pretty(cd)}" for p, cd in t)
            print(f"  trick: {plays}   -> {c(self.label(w), '1')} won")
        self.seen_tricks = len(obs.history)
        trump = (f"{pretty(obs.trump_card)} -> trump {SUIT_SYMBOLS[obs.trump]}" if obs.trump is not None
                 else f"{pretty(obs.trump_card)} -> no trump")
        print(c(f"\n── Round {obs.round_index}/{obs.n_rounds} · {obs.n_cards} card(s) · ", "1") + trump)
        for p in range(n):
            bid = "?" if obs.bids[p] is None else obs.bids[p]
            tag = " (dealer)" if p == obs.dealer else ""
            print(f"  {self.label(p) + tag:<30} bid {bid:>2}  won {obs.tricks_won[p]}   total {obs.game_scores[p]:>4}")
        if obs.trick:
            print("  on the table: " + "  ".join(f"{self.label(p)}: {pretty(cd)}" for p, cd in obs.trick))
            print(f"  currently winning: {self.label(trick_winner(obs.trick, obs.trump, obs.rules))}")
        self.shown_hand = sorted(obs.hand, key=hand_order)
        cells = []
        for i, cd in enumerate(self.shown_hand, 1):
            ok = obs.phase == "bid" or cd in obs.legal_actions
            cells.append(f"[{i}] {pretty(cd)}" if ok else c(f"[{i}] {card_str(cd)}", "2;9"))
        print("  your hand: " + "  ".join(cells))
        if obs.phase == "bid":
            taken = sum(b for b in obs.bids if b is not None)
            print(f"  bids so far total {taken}; allowed bids: {obs.legal_actions}")

    def hint(self, obs):
        print("  thinking...")
        root = ismcts(obs, 600)
        scale = 20 + 10 * obs.n_cards
        kids = sorted(root.children.items(), key=lambda kv: -kv[1].visits)[:3]
        total = sum(ch.visits for _, ch in root.children.items())
        for a, ch in kids:
            what = f"bid {a}" if obs.phase == "bid" else pretty(a)
            print(f"  AI likes {what:<10} ({100 * ch.visits / total:4.0f}% of search, "
                  f"expects ~{scale * ch.total / ch.visits:+.0f} pts this round)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seats", nargs="*", default=["human", "mcts:300", "heuristic", "heuristic"])
    ap.add_argument("--names", nargs="*", default=[])
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    n_humans = a.seats.count("human")
    names, bots, hi = [], [], 0
    for spec in a.seats:
        if spec == "human":
            nm = a.names[hi] if hi < len(a.names) else (f"Player {hi + 1}" if n_humans > 1 else "You")
            hi += 1
            bots.append(HumanBot(nm, names, hot_seat=n_humans > 1))
            names.append(nm)
        else:
            b = load_bot(spec)
            bots.append(b)
            names.append(f"{b.name} (bot)")
    # make bot names unique
    dup = [nm for nm in names if names.count(nm) > 1]
    for i, nm in enumerate(names):
        if nm in dup:
            names[i] = f"{nm} #{i + 1}"
    g = WizardGame(bots, seed=a.seed, n_rounds=a.rounds)
    orig = g.play_round

    def round_with_recap(r):
        orig(r)
        log = g.round_log[-1]
        print(c(f"\n== Round {r} results ==", "1;36"))
        for p in range(g.n):
            hit = "✓" if log["bids"][p] == log["won"][p] else "✗"
            print(f"  {names[p]:<22} bid {log['bids'][p]} won {log['won'][p]} {hit} {log['scores'][p]:+4}"
                  f"   total {g.scores[p]:>4}")
        if r < g.n_rounds:
            input("Enter for next round... ")
    g.play_round = round_with_recap
    final = g.play()
    print(c("\n🏆 Final standings", "1;33"))
    for i, p in enumerate(sorted(range(g.n), key=lambda p: -final[p]), 1):
        print(f"  {i}. {names[p]:<22} {final[p]}")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nbye!")
