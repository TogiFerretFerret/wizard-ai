"""Turn an Observation into a fixed-size vector for a neural net, and map actions to indices.

Action space (66): indices 0..13 = bids 0..13, indices 14..65 = cards 0..51.
Seats are encoded *relative to you* (0 = me, 1 = next player, ...) so the net is seat-invariant.
"""
import numpy as np

from engine import Observation

N_ACTIONS = 14 + 52
MAX_PLAYERS = 4


def action_index(obs: Observation, action: int) -> int:
    return action if obs.phase == "bid" else 14 + action


def index_action(obs: Observation, idx: int) -> int:
    return idx if obs.phase == "bid" else idx - 14


def legal_mask(obs: Observation) -> np.ndarray:
    m = np.zeros(N_ACTIONS, dtype=bool)
    m[[action_index(obs, a) for a in obs.legal_actions]] = True
    return m


def encode(obs: Observation) -> np.ndarray:
    n, me, nc = obs.n_players, obs.player, max(obs.n_cards, 1)
    rel = lambda p: (p - me) % n
    hand = np.zeros(52); hand[obs.hand] = 1
    played = np.zeros(52); played[obs.played] = 1
    trick = np.zeros((MAX_PLAYERS, 52))
    for p, c in obs.trick:
        trick[rel(p), c] = 1
    trump = np.zeros(5); trump[4 if obs.trump is None else obs.trump] = 1
    bids = np.full(MAX_PLAYERS, -1.0); won = np.zeros(MAX_PLAYERS); voids = np.zeros((MAX_PLAYERS, 4))
    for p in range(n):
        if obs.bids[p] is not None:
            bids[rel(p)] = obs.bids[p] / nc
        won[rel(p)] = obs.tricks_won[p] / nc
        for s in obs.voids[p]:
            voids[rel(p), s] = 1
    misc = np.array([obs.n_cards / 13, float(obs.phase == "bid"), rel(obs.dealer) / n,
                     len(obs.trick) / n, sum(b or 0 for b in obs.bids) / nc])
    return np.concatenate([hand, played, trick.ravel(), trump, bids, won, voids.ravel(), misc]).astype(np.float32)


FEATURE_SIZE = 52 + 52 + MAX_PLAYERS * 52 + 5 + 2 * MAX_PLAYERS + MAX_PLAYERS * 4 + 5
