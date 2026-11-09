"""Step 1 of the AlphaZero loop: generate training data from MCTS self-play.

For every decision we save
  x  = encode(obs)                       (what the player saw)
  pi = MCTS visit distribution           (policy target — "what search thought was best")
  z  = that player's final round score / (20 + 10*cards)   (value target)

Then train a net on (x -> pi, z), plug it into ismcts(value_fn=..., priors) and repeat
with the stronger search. Usage:  python selfplay.py --games 20 --iters 200 --out data.npz
"""
import argparse
import random

import numpy as np

from bots import ismcts
from engine import DECK, N_ROUNDS, RoundState
from features import N_ACTIONS, action_index, encode


def selfplay_round(n_cards, rng, iters, n_players=4):
    deck = list(DECK); rng.shuffle(deck)
    s = RoundState(n_players, n_cards, rng.randrange(n_players),
                   [deck[i * n_cards:(i + 1) * n_cards] for i in range(n_players)],
                   deck[n_players * n_cards] if n_players * n_cards < 52 else None)
    samples = []
    while not s.is_over():
        obs = s.observe(s.to_move)
        root = ismcts(obs, iters, rng=rng)
        pi = np.zeros(N_ACTIONS, dtype=np.float32)
        for a, ch in root.children.items():
            pi[action_index(obs, a)] = ch.visits
        pi /= pi.sum()
        samples.append((encode(obs), pi, obs.player))
        # sample proportionally early, play greedily later (AlphaZero's temperature trick)
        acts = list(root.children)
        a = rng.choices(acts, weights=[root.children[x].visits for x in acts])[0] \
            if len(samples) < 8 else max(acts, key=lambda x: root.children[x].visits)
        s.apply(a)
    scale = 20 + 10 * n_cards
    z = [x / scale for x in s.scores()]
    return [(x, pi, z[p]) for x, pi, p in samples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10, help="rounds of self-play per round size")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="selfplay.npz")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    data = []
    for g in range(a.games):
        for n_cards in range(1, N_ROUNDS + 1):
            data += selfplay_round(n_cards, rng, a.iters)
        print(f"game {g + 1}/{a.games}: {len(data)} samples")
    X, P, Z = map(np.stack, zip(*data))
    np.savez_compressed(a.out, x=X, pi=P, z=Z.astype(np.float32))
    print(f"saved {len(X)} samples to {a.out}  (x {X.shape}, pi {P.shape}, z {Z.shape})")


if __name__ == "__main__":
    main()
