"""Wizard game engine — club house rules.

Rules (from the club spec):
  * Standard 52-card deck. Aces = Wizards (always win), 2s = Jesters (always lose).
    3..K are standard cards in four suits.
  * Round r deals r cards to each player. A game is 10 rounds.
  * Next card of the deck sets trump. Wizard/Jester => no trump.
  * Bidding starts left of the dealer. "Screw the dealer": the dealer (last bidder)
    may not bid a number that makes total bids == cards dealt.
  * Player left of the dealer leads the first trick; trick winner leads the next.
  * Must follow the led suit if possible; Wizards and Jesters may always be played.
  * Trick winner: first Wizard > highest trump > highest card of led suit.
    If every card is a Jester, the LAST Jester wins (house rule).
    If a Jester leads, the first standard card sets the suit to follow.
  * Scoring: exact bid => 20 + 10 per trick. Miss => -10 per trick over/under.

Decisions the spec left open (configurable in `Rules`):
  * If a Wizard leads (or is the first non-Jester), there is no suit to follow.
  * Dealer rotates one seat left each round.

Cards are ints 0..51: suit = c // 13, rank = c % 13 + 2  (2 = Jester, 14 = Ace = Wizard).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

SUIT_CHARS = "SHDC"
SUIT_SYMBOLS = "♠♥♦♣"
RANK_CHARS = {11: "J", 12: "Q", 13: "K", 14: "A"}
DECK = tuple(range(52))
N_ROUNDS = 10  # club rule: a game is 10 rounds


def suit(c: int) -> int:
    return c // 13


def rank(c: int) -> int:
    return c % 13 + 2


def is_wizard(c: int) -> bool:
    return c % 13 == 12


def is_jester(c: int) -> bool:
    return c % 13 == 0


def is_standard(c: int) -> bool:
    return 0 < c % 13 < 12


def card_str(c: int | None) -> str:
    if c is None:
        return "--"
    if is_wizard(c):
        return "Wiz" + SUIT_SYMBOLS[suit(c)]
    if is_jester(c):
        return "Jes" + SUIT_SYMBOLS[suit(c)]
    r = rank(c)
    return RANK_CHARS.get(r, str(r)) + SUIT_SYMBOLS[suit(c)]


def make_card(rank_: int, suit_: int) -> int:
    return suit_ * 13 + rank_ - 2


@dataclass(frozen=True)
class Rules:
    last_jester_wins: bool = True   # all-Jester trick: last Jester wins (club rule)
    screw_the_dealer: bool = True   # dealer can't make bids sum to cards dealt


def lead_suit(trick: list[tuple[int, int]]) -> int | None:
    """Suit that must be followed, or None (nothing led yet / Wizard led)."""
    for _, c in trick:
        if is_jester(c):
            continue
        if is_wizard(c):
            return None
        return suit(c)
    return None


def trick_winner(trick: list[tuple[int, int]], trump: int | None,
                 rules: Rules = Rules()) -> int:
    """Return the player who wins `trick` (list of (player, card)) as it stands."""
    for p, c in trick:
        if is_wizard(c):
            return p
    standard = [(p, c) for p, c in trick if is_standard(c)]
    if not standard:
        return trick[-1][0] if rules.last_jester_wins else trick[0][0]
    if trump is not None:
        trumps = [(p, c) for p, c in standard if suit(c) == trump]
        if trumps:
            return max(trumps, key=lambda pc: rank(pc[1]))[0]
    led = lead_suit(trick)
    followers = [(p, c) for p, c in standard if suit(c) == led]
    return max(followers, key=lambda pc: rank(pc[1]))[0]


def round_score(bid: int, won: int) -> int:
    return 20 + 10 * won if bid == won else -10 * abs(bid - won)


class RoundState:
    """Full-information state of a single round. Cheap to copy — use it for search."""

    __slots__ = ("n_players", "n_cards", "dealer", "hands", "trump_card", "trump",
                 "bids", "tricks_won", "phase", "to_move", "trick", "history",
                 "voids", "rules")

    def __init__(self, n_players, n_cards, dealer, hands, trump_card, rules=Rules()):
        self.n_players = n_players
        self.n_cards = n_cards
        self.dealer = dealer
        self.hands = [list(h) for h in hands]
        self.trump_card = trump_card
        self.trump = suit(trump_card) if trump_card is not None and is_standard(trump_card) else None
        self.bids: list[int | None] = [None] * n_players
        self.tricks_won = [0] * n_players
        self.phase = "bid"
        self.to_move = (dealer + 1) % n_players
        self.trick: list[tuple[int, int]] = []
        self.history: list[tuple[list[tuple[int, int]], int]] = []  # (trick, winner)
        self.voids = [set() for _ in range(n_players)]  # public: suits a player showed out of
        self.rules = rules

    def copy(self) -> "RoundState":
        s = RoundState.__new__(RoundState)
        s.n_players, s.n_cards, s.dealer = self.n_players, self.n_cards, self.dealer
        s.hands = [h[:] for h in self.hands]
        s.trump_card, s.trump = self.trump_card, self.trump
        s.bids = self.bids[:]
        s.tricks_won = self.tricks_won[:]
        s.phase, s.to_move = self.phase, self.to_move
        s.trick = self.trick[:]
        s.history = self.history[:]  # tricks are never mutated after completion
        s.voids = [v.copy() for v in self.voids]
        s.rules = self.rules
        return s

    # ---- rules -------------------------------------------------------------
    def legal_actions(self) -> list[int]:
        if self.phase == "bid":
            options = list(range(self.n_cards + 1))
            if self.rules.screw_the_dealer and self.to_move == self.dealer:
                forbidden = self.n_cards - sum(b for b in self.bids if b is not None)
                if forbidden in options:
                    options.remove(forbidden)
            return options
        if self.phase == "play":
            hand = self.hands[self.to_move]
            led = lead_suit(self.trick)
            if led is not None:
                follow = [c for c in hand if is_standard(c) and suit(c) == led]
                if follow:
                    return follow + [c for c in hand if not is_standard(c)]
            return hand[:]
        return []

    def apply(self, action: int) -> None:
        p = self.to_move
        if self.phase == "bid":
            self.bids[p] = action
            if p == self.dealer:
                self.phase = "play"
                self.to_move = (self.dealer + 1) % self.n_players
            else:
                self.to_move = (p + 1) % self.n_players
            return
        c = action
        led = lead_suit(self.trick)
        if led is not None and is_standard(c) and suit(c) != led:
            self.voids[p].add(led)
        self.hands[p].remove(c)
        self.trick.append((p, c))
        if len(self.trick) < self.n_players:
            self.to_move = (p + 1) % self.n_players
            return
        w = trick_winner(self.trick, self.trump, self.rules)
        self.tricks_won[w] += 1
        self.history.append((self.trick, w))
        self.trick = []
        self.to_move = w
        if not self.hands[w]:
            self.phase = "done"

    def is_over(self) -> bool:
        return self.phase == "done"

    def scores(self) -> list[int]:
        return [round_score(b, w) for b, w in zip(self.bids, self.tricks_won)]

    def observe(self, player: int, **extra) -> "Observation":
        played = [c for t, _ in self.history for _, c in t] + [c for _, c in self.trick]
        return Observation(
            player=player, n_players=self.n_players, n_cards=self.n_cards,
            dealer=self.dealer, phase=self.phase, hand=sorted(self.hands[player]),
            trump_card=self.trump_card, trump=self.trump, bids=self.bids[:],
            tricks_won=self.tricks_won[:], trick=self.trick[:],
            history=[(t[:], w) for t, w in self.history], played=played,
            voids=[set(v) for v in self.voids],
            legal_actions=self.legal_actions() if self.to_move == player else [],
            rules=self.rules, **extra)


@dataclass
class Observation:
    """Everything one player is allowed to see. This is what your bot receives."""
    player: int
    n_players: int
    n_cards: int            # cards dealt this round (= round number)
    dealer: int
    phase: str              # "bid" or "play"
    hand: list[int]
    trump_card: int | None
    trump: int | None       # trump suit, None if no trump
    bids: list[int | None]
    tricks_won: list[int]
    trick: list[tuple[int, int]]          # current trick so far: (player, card)
    history: list[tuple[list, int]]       # finished tricks this round: (trick, winner)
    played: list[int]                     # every card played this round
    voids: list[set[int]]                 # suits each player has shown out of
    legal_actions: list[int]              # bids (ints) or cards (ints)
    rules: Rules = field(default_factory=Rules)
    game_scores: list[int] = field(default_factory=list)  # totals before this round
    round_index: int = 0
    n_rounds: int = 0

    def unseen_cards(self) -> list[int]:
        """Cards that could be in an opponent's hand (or undealt)."""
        known = set(self.hand) | set(self.played)
        if self.trump_card is not None:
            known.add(self.trump_card)
        return [c for c in DECK if c not in known]

    def hand_sizes(self) -> list[int]:
        played_by = [0] * self.n_players
        for t, _ in self.history:
            for p, _c in t:
                played_by[p] += 1
        for p, _c in self.trick:
            played_by[p] += 1
        return [self.n_cards - k for k in played_by]


def determinize(obs: Observation, rng: random.Random | None = None,
                tries: int = 200) -> RoundState:
    """Sample a full RoundState consistent with what `obs.player` can see.

    Opponent hands are random, but respect known voids (a player who failed to
    follow suit can't hold standard cards of that suit). This is the key trick for
    running MCTS in a hidden-information game.
    """
    rng = rng or random
    me = obs.player
    sizes = obs.hand_sizes()
    unseen = obs.unseen_cards()
    others = [p for p in range(obs.n_players) if p != me]

    def allowed(p, c):
        return not (is_standard(c) and suit(c) in obs.voids[p])

    hands = None
    for _ in range(tries):
        rng.shuffle(unseen)
        trial = {p: [] for p in others}
        used = set()
        # place most-constrained players first
        ok = True
        for p in sorted(others, key=lambda q: -len(obs.voids[q])):
            for c in unseen:
                if len(trial[p]) == sizes[p]:
                    break
                if c not in used and allowed(p, c):
                    trial[p].append(c)
                    used.add(c)
            if len(trial[p]) < sizes[p]:
                ok = False
                break
        if ok:
            hands = trial
            break
    if hands is None:  # constraints too tight for random sampling; ignore them
        rng.shuffle(unseen)
        hands, i = {}, 0
        for p in others:
            hands[p] = unseen[i:i + sizes[p]]
            i += sizes[p]
    hands[me] = list(obs.hand)

    s = RoundState(obs.n_players, obs.n_cards, obs.dealer,
                   [hands[p] for p in range(obs.n_players)], obs.trump_card, obs.rules)
    s.bids = obs.bids[:]
    s.tricks_won = obs.tricks_won[:]
    s.phase = obs.phase
    s.trick = obs.trick[:]
    s.history = [(t[:], w) for t, w in obs.history]
    s.voids = [set(v) for v in obs.voids]
    s.to_move = obs.player
    return s


class Bot:
    """Subclass this. `act` gets an Observation and returns one of obs.legal_actions.

    In the bid phase return an int bid; in the play phase return a card int.
    """
    name = "bot"

    def act(self, obs: Observation) -> int:
        raise NotImplementedError

    def new_game(self, seat: int, n_players: int) -> None:
        """Optional hook called before each game."""

    def round_over(self, final_state: "RoundState", round_scores: list[int],
                   totals_before: list[int]) -> None:
        """Optional hook called after each round (all hands are revealed in final_state.history)."""


class WizardGame:
    def __init__(self, bots: list[Bot], seed: int | None = None, rules: Rules = Rules(),
                 n_rounds: int | None = None, verbose: bool = False):
        assert 3 <= len(bots) <= 6
        self.bots = bots
        self.n = len(bots)
        self.rng = random.Random(seed)
        self.rules = rules
        self.n_rounds = min(n_rounds or N_ROUNDS, 52 // self.n)
        self.verbose = verbose
        self.scores = [0] * self.n
        self.round_log: list[dict] = []
        self.think_time = [0.0] * self.n   # seconds each seat spent deciding
        self.decisions = [0] * self.n

    def log(self, *a):
        if self.verbose:
            print(*a)

    def play(self) -> list[int]:
        for seat, b in enumerate(self.bots):
            b.new_game(seat, self.n)
        for r in range(1, self.n_rounds + 1):
            self.play_round(r)
        return self.scores

    def play_round(self, r: int) -> None:
        dealer = (r - 1) % self.n
        deck = list(DECK)
        self.rng.shuffle(deck)
        hands = [deck[i * r:(i + 1) * r] for i in range(self.n)]
        trump_card = deck[self.n * r] if self.n * r < 52 else None
        s = RoundState(self.n, r, dealer, hands, trump_card, self.rules)
        self.log(f"\n=== Round {r}  dealer P{dealer}  trump card {card_str(trump_card)} "
                 f"({SUIT_SYMBOLS[s.trump] if s.trump is not None else 'no trump'}) ===")
        for p in range(self.n):
            self.log(f"  P{p} [{self.bots[p].name}]: " + " ".join(card_str(c) for c in sorted(hands[p])))
        while not s.is_over():
            p = s.to_move
            obs = s.observe(p, game_scores=self.scores[:], round_index=r,
                            n_rounds=self.n_rounds)
            t0 = time.perf_counter()
            a = self.bots[p].act(obs)
            self.think_time[p] += time.perf_counter() - t0
            self.decisions[p] += 1
            if a not in obs.legal_actions:
                raise ValueError(f"{self.bots[p].name} (P{p}) made illegal move {a!r} "
                                 f"in phase {s.phase}; legal: {obs.legal_actions}")
            if s.phase == "bid":
                self.log(f"  P{p} bids {a}")
            else:
                self.log(f"  P{p} plays {card_str(a)}" + ("" if len(s.trick) < self.n - 1 else
                         f"  -> P{trick_winner(s.trick + [(p, a)], s.trump, self.rules)} wins"))
            s.apply(a)
        rs = s.scores()
        for b in self.bots:
            b.round_over(s, rs, self.scores[:])
        for p in range(self.n):
            self.scores[p] += rs[p]
        self.round_log.append({"round": r, "bids": s.bids, "won": s.tricks_won, "scores": rs})
        self.log(f"  bids {s.bids} won {s.tricks_won} -> {rs}   totals {self.scores}")
