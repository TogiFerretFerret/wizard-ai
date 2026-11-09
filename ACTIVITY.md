# 🧙 Club Activity Guide: Wizard AI

A four-meeting plan, about 45–60 min each. Everything runs with plain Python 3.10+. Only the neural-net part (meeting 4) needs `pip install numpy torch`.

```bash
git clone https://github.com/TogiFerretFerret/wizard-ai && cd wizard-ai
```

---

## Meeting 1: Learn the game by playing it (and losing to a bot)

**Goal:** everyone knows the rules and has an intuition for *why* bidding is hard.

1. **Warm-up (10 min).** Pass-and-play one short game together on one laptop:
   `python play.py human human human human --names Ana Ben Cy Dee --rounds 4`
2. **Human vs. AI (15 min).** Everyone plays a game against the bots:
   `python play.py` (you vs. MCTS + 2 rule-based bots).
   Type `?` on any turn to see what the AI would do, and how many points it expects.
3. **Watch bots play (5 min).** `python tournament.py mcts:200 heuristic heuristic random --watch --rounds 4`
4. **Discuss (10 min).**
   - What did you consider when you bid? Write it down. That's your first algorithm.
   - When did the AI's `?` hint surprise you?
   - Screw-the-dealer: is being the dealer good or bad?

**Homework:** copy `players/_template.py` to `players/<yourname>.py`.

---

## Meeting 2: Write a rule-based bot and climb the ladder

**Goal:** everyone has a bot that beats random play. The keen ones beat the heuristic bot.

1. Read `players/_template.py` and `players/example_greedy.py` (5 min). A bot is one method, `act(obs)`, and it returns a bid or a card from `obs.legal_actions`.
2. Code (25 min). Turn the rules you wrote down last week into Python. Test it often:
   `python challenge.py players/<yourname>.py`

   | level | opponents | |
   |---|---|---|
   | 🃏 Apprentice | 3 × random | almost anything sensible passes |
   | 📜 Journeyman | 3 × heuristic | needs decent bidding *and* ducking |
   | 🔮 Adept | MCTS-50 + 2 heuristic | needs lookahead or very good rules |
   | 🧙 Archmage | 3 × MCTS-200 | the real challenge |

3. **League (10 min).** Push or AirDrop your files into `players/`, then run `python league.py` on one machine. It writes `LEADERBOARD.md`. Put it on the projector.

Ideas to steal from each other:
- Count trumps and Wizards for the bid. Don't bid on low cards when you hold many cards.
- When you've made your bid, *duck*: play the highest card that still loses.
- Track `obs.voids`: who has run out of which suit?
- Lead your Wizard early if you need tricks. Hold Jesters as escape cards.

---

## Meeting 3: Monte Carlo Tree Search

**Goal:** understand why search beats hand-written rules, then make a search bot of your own.

Concepts (whiteboard, 15 min):
1. **Simulation.** Imagine the rest of the round many times and pick the move that worked best on average.
2. **The hidden-info problem.** We can't see the other hands. So we *guess* them (`determinize(obs)`): deal the unseen cards randomly, but respect what we know. If Ben didn't follow ♥, Ben has no ♥.
3. **UCB / tree search.** Spend more simulations on promising moves, but keep exploring the others (`bots.py → ismcts`).

Hands-on (30 min), with a starting point for each person:
- Use `determinize(obs)` in your own bot. Try each legal move on 50 guessed deals and pick the best. That's "flat Monte Carlo", about 20 lines.
- Tune MCTS: compare `mcts:50` vs `mcts:400` with `python tournament.py ...`. Is more thinking always better?
- Swap the rollout policy from `heuristic` to `random`. How much worse does it get, and why?
- Make your bot's bidding use search, but keep your own play rules.

---

## Meeting 4: Learning (the AlphaZero part)

**Goal:** see a neural net learn from its own games.

AlphaZero loop:
```
   self-play with search  ──►  data: (what I saw, what search preferred, how it ended)
            ▲                                │
            │                                ▼
   better search  ◄── net guides search ◄── train policy + value net
```

1. Run training on the strongest computer (GPU if available):
   ```bash
   python -m venv .venv && .venv/bin/pip install numpy torch
   .venv/bin/python train.py --iters 6 --games 64 --sims 100
   ```
   Every iteration prints a table: NetBot vs plain MCTS with the same thinking budget. Watch the gap close.
2. While it trains, discuss:
   - Policy head = "which moves look good", which lets search skip bad moves.
   - Value head = "how many points I'll get from here", which replaces random rollouts.
   - Why might this be harder in Wizard than in chess? (hidden cards, 4 players, luck)
3. Add `netbot.py` to the league. Can anyone's hand-written bot still beat it?

---

## Tournament day

- Freeze bots at a deadline. Run `python league.py --tables 60 --deals 3` on the fastest machine.
- **Time limit:** keep `ms/move` under ~200 so a full league finishes in a few minutes.
- Prizes: 🥇 top margin, 🎯 best bidder (most exact bids), 🤡 most creative losing strategy.
- Final boss: the league winner vs. the club's best human player, live with `python play.py human players/winner.py ...`.

## Fair-play rules for bots

- Use only `obs`. Don't read other players' hands from the game object.
- No randomness tricks with the global seed. Use your own `random.Random()`.
- One file in `players/`. Helper code goes in the same file.
