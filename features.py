"""Turn an Observation into a fixed-size vector for a neural net, and map actions to indices.

Action space (66): indices 0..13 = bids 0..13 (only 0..10 used in a 10-round game), indices 14..65 = cards 0..51.
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
    misc = np.array([obs.n_cards / 10, float(obs.phase == "bid"), rel(obs.dealer) / n,
                     len(obs.trick) / n, sum(b or 0 for b in obs.bids) / nc])
    return np.concatenate([hand, played, trick.ravel(), trump, bids, won, voids.ravel(), misc]).astype(np.float32)


FEATURE_SIZE = 52 + 52 + MAX_PLAYERS * 52 + 5 + 2 * MAX_PLAYERS + MAX_PLAYERS * 4 + 5


# Offsets into the feature vector (must match encode()).
_HAND, _PLAYED, _TRICK = 0, 52, 104
_TRUMP = _TRICK + MAX_PLAYERS * 52
_BIDS = _TRUMP + 5
_WON = _BIDS + MAX_PLAYERS
_VOIDS = _WON + MAX_PLAYERS
_MISC = _VOIDS + MAX_PLAYERS * 4


def encode_state(s, me: int) -> np.ndarray:
    """Same vector as encode(s.observe(me)), but ~10x faster (used inside search)."""
    n, nc = s.n_players, max(s.n_cards, 1)
    x = np.zeros(FEATURE_SIZE, np.float32)
    x[s.hands[me]] = 1
    taken = 0
    for t, _w in s.history:
        for _p, c in t:
            x[_PLAYED + c] = 1
    for p, c in s.trick:
        x[_PLAYED + c] = 1
        x[_TRICK + ((p - me) % n) * 52 + c] = 1
    x[_TRUMP + (4 if s.trump is None else s.trump)] = 1
    x[_BIDS:_BIDS + MAX_PLAYERS] = -1
    for p in range(n):
        r = (p - me) % n
        b = s.bids[p]
        if b is not None:
            x[_BIDS + r] = b / nc
            taken += b
        x[_WON + r] = s.tricks_won[p] / nc
        for su in s.voids[p]:
            x[_VOIDS + r * 4 + su] = 1
    x[_MISC:_MISC + 5] = (s.n_cards / 10, float(s.phase == "bid"), ((s.dealer - me) % n) / n,
                          len(s.trick) / n, taken / nc)
    return x
