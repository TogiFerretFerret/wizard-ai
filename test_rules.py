"""Run: python test_rules.py   (checks the club rules are implemented as written)"""
import random
from engine import RoundState, Rules, make_card as C, trick_winner, determinize, WizardGame
from bots import RandomBot

S, H, D, Cl = 0, 1, 2, 3
W = lambda s: C(14, s)   # Wizard (Ace)
J = lambda s: C(2, s)    # Jester (Two)

# first wizard wins
assert trick_winner([(0, C(13, S)), (1, W(H)), (2, W(S)), (3, C(5, D))], trump=D) == 1
# highest trump beats led suit
assert trick_winner([(0, C(13, S)), (1, C(3, D)), (2, C(5, D)), (3, C(12, S))], trump=D) == 2
# no trump played: highest of led suit; off-suit high card loses
assert trick_winner([(0, C(5, S)), (1, C(13, H)), (2, C(9, S)), (3, C(3, S))], trump=D) == 2
# no trump at all
assert trick_winner([(0, C(5, S)), (1, C(13, H)), (2, C(9, S))], trump=None) == 2
# all jesters: LAST jester wins (club rule)
assert trick_winner([(0, J(S)), (1, J(H)), (2, J(D)), (3, J(Cl))], trump=S) == 3
assert trick_winner([(0, J(S)), (1, J(H))], trump=S, rules=Rules(last_jester_wins=False)) == 0
# jester lead: next standard card sets the suit
assert trick_winner([(0, J(S)), (1, C(4, H)), (2, C(13, S)), (3, C(9, H))], trump=None) == 3

# follow suit, wizards/jesters always allowed
s = RoundState(4, 3, 3, [[C(5, H), C(9, S), W(D)], [C(4, H), J(Cl), C(10, Cl)],
                        [C(3, S), C(6, S), C(7, S)], [C(3, D), C(4, D), C(5, D)]], C(8, Cl))
for b in (1, 1, 0):
    s.apply(b)
assert 1 not in s.legal_actions()   # screw the dealer: 1+1+0+1 == 3 forbidden
s.apply(0)
s.apply(C(5, H))                         # P0 leads hearts
assert sorted(s.legal_actions()) == sorted([C(4, H), J(Cl)])
s.apply(J(Cl))
assert sorted(s.legal_actions()) == [C(3, S), C(6, S), C(7, S)]   # P2 void in hearts
s.apply(C(7, S))
assert H in s.voids[2]
# wizard led -> anything goes
s2 = RoundState(4, 2, 3, [[W(S), C(3, H)], [C(4, H), C(5, S)], [C(6, H), C(7, D)], [C(8, H), C(9, D)]], None)
for b in (1, 0, 0, 0):
    s2.apply(b)
s2.apply(W(S))
assert sorted(s2.legal_actions()) == sorted([C(4, H), C(5, S)])

# determinization respects voids and hand sizes
rng = random.Random(1)
for _ in range(300):
    deck = list(range(52)); rng.shuffle(deck)
    st = RoundState(4, 8, 0, [deck[i*8:(i+1)*8] for i in range(4)], deck[32])
    while not st.is_over() and rng.random() < 0.97:
        st.apply(rng.choice(st.legal_actions()))
    if st.is_over():
        continue
    d = determinize(st.observe(st.to_move), rng)
    assert [len(h) for h in d.hands] == [len(h) for h in st.hands]
    for p in range(4):
        for c in d.hands[p]:
            assert not (0 < c % 13 < 12 and c // 13 in st.voids[p])

# full games run clean
for seed in range(20):
    WizardGame([RandomBot(seed + i) for i in range(4)], seed=seed).play()
print("all rule tests passed")
