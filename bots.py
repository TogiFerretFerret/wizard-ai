"""Baseline bots: Random, Heuristic, and ISMCTS (Monte Carlo tree search for hidden info)."""
from __future__ import annotations

import math
import random

from engine import (Bot, Observation, RoundState, determinize, is_jester, is_standard,
                    is_wizard, rank, suit, trick_winner)


class RandomBot(Bot):
    name = "random"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, obs):
        return self.rng.choice(obs.legal_actions)


# --------------------------------------------------------------------------- heuristic
def heuristic_bid(hand, trump, n_players, legal):
    """Count 'sure-ish' tricks: wizards, high trumps, off-suit kings."""
    est = 0.0
    for c in hand:
        if is_wizard(c):
            est += 1.0
        elif is_standard(c):
            r = rank(c)
            if trump is not None and suit(c) == trump:
                est += 0.9 if r >= 11 else 0.5 if r >= 8 else 0.2
            elif r == 13:
                est += 0.6
            elif r == 12:
                est += 0.3 if len(hand) > 3 else 0.15
    target = round(est)
    return min(legal, key=lambda b: (abs(b - target), b))


def heuristic_play(hand_legal, trick, trump, me, want_trick, rules):
    """want_trick: True if I still need tricks. Win cheaply, or duck high."""
    def wins(c):
        return trick_winner(trick + [(me, c)], trump, rules) == me

    def strength(c):  # rough card power for tie-breaking
        if is_wizard(c):
            return 100
        if is_jester(c):
            return -1
        return rank(c) + (20 if trump is not None and suit(c) == trump else 0)

    winners = [c for c in hand_legal if wins(c)]
    losers = [c for c in hand_legal if c not in winners]
    if want_trick:
        if winners:
            # leading: play strongest; otherwise win as cheaply as possible
            return max(winners, key=strength) if not trick else min(winners, key=strength)
        return min(hand_legal, key=strength)
    if losers:
        return max(losers, key=strength)  # dump the most dangerous card that still loses
    return min(hand_legal, key=strength)


class HeuristicBot(Bot):
    name = "heuristic"

    def act(self, obs):
        if obs.phase == "bid":
            return heuristic_bid(obs.hand, obs.trump, obs.n_players, obs.legal_actions)
        me = obs.player
        want = obs.tricks_won[me] < obs.bids[me]
        return heuristic_play(obs.legal_actions, obs.trick, obs.trump, me, want, obs.rules)


def heuristic_rollout_action(s: RoundState, rng: random.Random) -> int:
    """Fast heuristic policy on a full state (only uses the mover's own info)."""
    legal = s.legal_actions()
    if rng.random() < 0.1:  # a little noise keeps rollouts diverse
        return rng.choice(legal)
    p = s.to_move
    if s.phase == "bid":
        return heuristic_bid(s.hands[p], s.trump, s.n_players, legal)
    return heuristic_play(legal, s.trick, s.trump, p, s.tricks_won[p] < s.bids[p], s.rules)


def random_rollout_action(s: RoundState, rng: random.Random) -> int:
    return rng.choice(s.legal_actions())


# --------------------------------------------------------------------------- ISMCTS
class Node:
    __slots__ = ("parent", "action", "player", "children", "visits", "avail", "total")

    def __init__(self, parent, action, player):
        self.parent = parent
        self.action = action
        self.player = player       # who chose `action` to reach this node
        self.children = {}
        self.visits = 0
        self.avail = 1
        self.total = 0.0           # summed reward for `player`


def ismcts(obs: Observation, iterations: int, c: float = 0.7, rollout=heuristic_rollout_action,
           rng: random.Random | None = None, value_fn=None) -> Node:
    """Single-Observer Information-Set MCTS (Cowling, Powley & Whitehouse 2012).

    Each iteration: sample a determinization (guess everyone's hidden cards), walk
    the shared tree using only moves legal in that guess, expand one node, roll out
    to the end of the round, and back up each player's own round score.

    `value_fn(state) -> list[float]` (optional) replaces the rollout with a learned
    evaluation — this is where an AlphaZero-style value network plugs in.
    """
    rng = rng or random.Random()
    root = Node(None, None, None)
    for _ in range(iterations):
        s = determinize(obs, rng)
        node = root
        # selection
        while not s.is_over():
            legal = s.legal_actions()
            untried = [a for a in legal if a not in node.children]
            for a in legal:
                ch = node.children.get(a)
                if ch is not None:
                    ch.avail += 1
            if untried:
                break
            best, best_v = None, -1e9
            for a in legal:
                ch = node.children[a]
                v = ch.total / ch.visits + c * math.sqrt(math.log(ch.avail) / ch.visits)
                if v > best_v:
                    best, best_v = ch, v
            node = best
            s.apply(node.action)
        # expansion
        if not s.is_over():
            a = rng.choice(untried)
            child = Node(node, a, s.to_move)
            node.children[a] = child
            s.apply(a)
            node = child
        # simulation
        if value_fn is not None and not s.is_over():
            rewards = value_fn(s)
        else:
            while not s.is_over():
                s.apply(rollout(s, rng))
            scale = 20 + 10 * s.n_cards
            rewards = [x / scale for x in s.scores()]
        # backpropagation
        while node is not None:
            node.visits += 1
            if node.player is not None:
                node.total += rewards[node.player]
            node = node.parent
    return root


class MCTSBot(Bot):
    def __init__(self, iterations=300, c=0.7, rollout="heuristic", seed=None):
        self.iterations = iterations
        self.c = c
        self.rollout = heuristic_rollout_action if rollout == "heuristic" else random_rollout_action
        self.rng = random.Random(seed)
        self.name = f"mcts{iterations}"

    def act(self, obs):
        if len(obs.legal_actions) == 1:
            return obs.legal_actions[0]
        root = ismcts(obs, self.iterations, self.c, self.rollout, self.rng)
        return max(obs.legal_actions,
                   key=lambda a: root.children[a].visits if a in root.children else -1)
