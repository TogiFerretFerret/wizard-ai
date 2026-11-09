"""AlphaZero-style bot: ISMCTS guided by a policy/value network.

The network is a small MLP trained by train.py (on GPU if you have one). Here it runs in
plain numpy so that self-play can use every CPU core without fighting over the GPU.

  python tournament.py netbot.py heuristic heuristic mcts:200
  NETBOT_MODEL=models/iter_005.npz NETBOT_SIMS=400 python tournament.py netbot.py ...
"""
from __future__ import annotations

import math
import os
import random

import numpy as np

from engine import Bot, Observation, RoundState, determinize
from features import N_ACTIONS, encode

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "latest.npz")


class NumpyNet:
    """Forward pass of train.PolicyValueNet using exported weights."""

    def __init__(self, path):
        w = np.load(path)
        self.layers = []
        i = 0
        while f"body.{i}.weight" in w:
            self.layers.append((w[f"body.{i}.weight"].T.copy(), w[f"body.{i}.bias"]))
            i += 2  # Linear, ReLU, Linear, ReLU, ...
        self.pw, self.pb = w["policy.weight"].T.copy(), w["policy.bias"]
        self.vw, self.vb = w["value.weight"].T.copy(), w["value.bias"]

    def __call__(self, x: np.ndarray):
        h = x
        for W, b in self.layers:
            h = np.maximum(h @ W + b, 0)
        return h @ self.pw + self.pb, (h @ self.vw + self.vb)[:, 0]


def _action_idx(phase, a):
    return a if phase == "bid" else 14 + a


class PNode:
    __slots__ = ("children", "priors", "N", "W", "player")

    def __init__(self, player=None):
        self.children = {}
        self.priors = None
        self.N = 0
        self.W = 0.0
        self.player = player


def evaluate(net, s: RoundState):
    """Policy priors for the mover's legal moves + predicted value for every player."""
    n = s.n_players
    X = np.stack([encode(s.observe(p)) for p in range(n)])
    logits, values = net(X)
    legal = s.legal_actions()
    lg = np.array([logits[s.to_move, _action_idx(s.phase, a)] for a in legal])
    lg = np.exp(lg - lg.max())
    lg /= lg.sum()
    return dict(zip(legal, lg)), values


def net_search(obs: Observation, net, sims: int = 100, c_puct: float = 1.5,
               rng: random.Random | None = None, root_noise: float = 0.0) -> PNode:
    """PUCT search over determinizations (AlphaZero selection + ISMCTS sampling)."""
    rng = rng or random.Random()
    root = PNode()
    for _ in range(sims):
        s = determinize(obs, rng)
        node, path = root, [root]
        while True:
            if s.is_over():
                scale = 20 + 10 * s.n_cards
                values = [x / scale for x in s.scores()]
                break
            legal = s.legal_actions()
            if node.priors is None or any(a not in node.priors for a in legal):
                pri, values = evaluate(net, s)
                first = node.priors is None
                if first:
                    node.priors = pri
                    if node is root and root_noise > 0:
                        noise = np.random.dirichlet([0.3] * len(pri))
                        for (a, p), e in zip(list(pri.items()), noise):
                            pri[a] = (1 - root_noise) * p + root_noise * e
                    break
                for a in legal:
                    node.priors.setdefault(a, pri[a])
            mover = s.to_move
            sqrt_n = math.sqrt(sum(node.children[a].N for a in legal if a in node.children) + 1)
            best, best_v = None, -1e9
            for a in legal:
                ch = node.children.get(a)
                q, n_ = (ch.W / ch.N, ch.N) if ch and ch.N else (0.0, 0)
                v = q + c_puct * node.priors[a] * sqrt_n / (1 + n_)
                if v > best_v:
                    best, best_v = a, v
            ch = node.children.get(best)
            if ch is None:
                ch = node.children[best] = PNode(mover)
            s.apply(best)
            node = ch
            path.append(node)
        for nd in path:
            nd.N += 1
            if nd.player is not None:
                nd.W += values[nd.player]
    return root


class NetBot(Bot):
    def __init__(self, model=None, sims=None, seed=None):
        model = model or os.environ.get("NETBOT_MODEL", DEFAULT_MODEL)
        self.net = NumpyNet(model)
        self.sims = sims or int(os.environ.get("NETBOT_SIMS", 100))
        self.rng = random.Random(seed)
        self.name = f"netbot{self.sims}"

    def act(self, obs):
        if len(obs.legal_actions) == 1:
            return obs.legal_actions[0]
        root = net_search(obs, self.net, self.sims, rng=self.rng)
        return max(obs.legal_actions, key=lambda a: root.children[a].N if a in root.children else -1)
