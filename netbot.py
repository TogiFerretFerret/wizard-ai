"""AlphaZero-style bot: ISMCTS guided by a policy/value network.

The network is an MLP trained by train.py (on the GPU if you have one). Here it runs in
plain numpy so self-play can use every CPU core.

  python tournament.py netbot.py heuristic heuristic mcts:200
  NETBOT_MODEL=models/iter_005.npz NETBOT_SIMS=400 python tournament.py netbot.py ...
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # tiny matmuls: threads only hurt, esp. with many workers

import math
import random

import numpy as np

from bots import heuristic_rollout_action
from engine import Bot, Observation, RoundState, determinize
from features import encode_state

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")


def default_model():
    for name in ("best.npz", "latest.npz"):
        p = os.path.join(MODELS, name)
        if os.path.exists(p):
            return p
    raise SystemExit("no trained model in models/ - run train.py first")


class NumpyNet:
    """Forward pass of train.PolicyValueNet using exported weights.

    Input: one encoded view per row. Output per row: policy logits (66) and the predicted
    normalized round score of the player whose view it is.
    """

    def __init__(self, path):
        w = np.load(path)
        self.layers = []
        i = 0
        while f"body.{i}.weight" in w:
            self.layers.append((np.ascontiguousarray(w[f"body.{i}.weight"].T), w[f"body.{i}.bias"]))
            i += 2  # Linear, ReLU, Linear, ReLU, ...
        self.pw, self.pb = np.ascontiguousarray(w["policy.weight"].T), w["policy.bias"]
        self.vw, self.vb = np.ascontiguousarray(w["value.weight"].T), w["value.bias"]
        self.mix = float(w["meta_mix"]) if "meta_mix" in w else 0.0

    def __call__(self, x: np.ndarray):
        h = x
        for W, b in self.layers:
            h = np.maximum(h @ W + b, 0)
        return h @ self.pw + self.pb, h @ self.vw + self.vb


def _norm_scores(s: RoundState):
    scale = 20 + 10 * s.n_cards
    return [x / scale for x in s.scores()]


class PNode:
    __slots__ = ("children", "priors", "expo", "N", "W", "player")

    def __init__(self, player=None):
        self.children = {}
        self.priors = None   # prior per action seen so far
        self.expo = None     # exp(logits) over the full action space, cached
        self.N = 0
        self.W = 0.0
        self.player = player


def evaluate(net, s: RoundState, mix: float = 0.0, rng=None):
    """Priors over the mover's legal moves + value for every player (absolute seats).

    mix > 0 blends the net's value with one heuristic playout (helps young nets).
    """
    me, n = s.to_move, s.n_players
    # each player's value must come from *their own* view (only they know their hand)
    out_logits, v = net(np.stack([encode_state(s, p) for p in range(n)]))
    logits = out_logits[me]
    off = 0 if s.phase == "bid" else 14
    legal = s.legal_actions()
    expo = np.exp(logits - logits.max()).tolist()
    expo = expo[:14] if off == 0 else expo[14:]   # index by bid value or by card id
    z = sum(expo[a] for a in legal)
    pri = {a: expo[a] / z for a in legal}
    values = v[:, 0].tolist()
    if mix > 0:
        sim = s.copy()
        while not sim.is_over():
            sim.apply(heuristic_rollout_action(sim, rng or random))
        r = _norm_scores(sim)
        values = [(1 - mix) * a + mix * b for a, b in zip(values, r)]
    return pri, values, expo


def net_search(obs: Observation, net, sims: int = 100, c_puct: float = 1.5,
               rng: random.Random | None = None, root_noise: float = 0.0,
               mix: float | None = None) -> PNode:
    """PUCT search over determinizations (AlphaZero selection + ISMCTS sampling)."""
    rng = rng or random.Random()
    mix = net.mix if mix is None else mix
    root = PNode()
    for _ in range(sims):
        s = determinize(obs, rng)
        node, path = root, [root]
        while True:
            if s.is_over():
                values = _norm_scores(s)
                break
            legal = s.legal_actions()
            if node.priors is None:
                node.priors, values, node.expo = evaluate(net, s, mix, rng)
                if node is root and root_noise > 0:
                    pri = node.priors
                    noise = np.random.dirichlet([0.3] * len(pri))
                    for (a, p), e in zip(list(pri.items()), noise):
                        pri[a] = (1 - root_noise) * p + root_noise * e
                break
            pri = node.priors
            if any(a not in pri for a in legal):  # moves only legal in this determinization
                z = sum(node.expo[a] for a in pri)
                for a in legal:
                    if a not in pri:
                        pri[a] = node.expo[a] / z
            mover = s.to_move
            sqrt_n = math.sqrt(sum(node.children[a].N for a in legal if a in node.children) + 1)
            best, best_v = None, -1e9
            for a in legal:
                ch = node.children.get(a)
                q, n_ = (ch.W / ch.N, ch.N) if ch is not None and ch.N else (0.0, 0)
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
    def __init__(self, model=None, sims=None, mix=None, seed=None):
        model = model or os.environ.get("NETBOT_MODEL") or default_model()
        self.net = NumpyNet(model)
        self.sims = sims or int(os.environ.get("NETBOT_SIMS", 200))
        env_mix = os.environ.get("NETBOT_MIX")
        self.mix = mix if mix is not None else (float(env_mix) if env_mix else None)
        self.rng = random.Random(seed)
        self.name = f"netbot{self.sims}"

    def act(self, obs):
        if len(obs.legal_actions) == 1:
            return obs.legal_actions[0]
        root = net_search(obs, self.net, self.sims, rng=self.rng, mix=self.mix)
        return max(obs.legal_actions, key=lambda a: root.children[a].N if a in root.children else -1)
