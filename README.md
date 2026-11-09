# Wizard AI Arena

Build an AI that plays **Wizard** (club house rules), then pit everyone's bots against each other.

The plan is roughly **AlphaZero for a hidden-information card game**: Monte Carlo tree search, then reinforcement learning to make the search smarter.

**Running a club session? Start with [ACTIVITY.md](ACTIVITY.md).** It's a four-meeting plan.

## Quick start (Python 3.10+, no installs needed except for the neural net)

```bash
python play.py                                  # play against the bots (type ? for an AI hint)
python play.py human human human human --names Ana Ben Cy Dee   # pass-and-play
cp players/_template.py players/alice.py        # write your own bot
python challenge.py players/alice.py            # climb the 4-level ladder
python league.py                                # everyone in players/ vs everyone -> LEADERBOARD.md
python tournament.py players/alice.py mcts:200 heuristic heuristic   # one custom table
python tournament.py mcts:100 heuristic heuristic random --watch --rounds 3   # watch every move
```

## Files

| file | what it is |
|---|---|
| `play.py` | Play in the terminal against bots or friends, with `?` hints from MCTS |
| `challenge.py` | Ladder: Apprentice → Journeyman → Adept → Archmage |
| `league.py` | Round-robin of every bot in `players/`. Ranked by score margin vs table average |
| `tournament.py` | One table of 3–6 bots, seat-rotated |
| `players/` | Put your bot here. `_template.py` to copy, `example_greedy.py` to beat |
| `engine.py` | Rules, `RoundState` (copyable full state for search), `Observation` (what a bot sees), `determinize()` |
| `bots.py` | `RandomBot`, `HeuristicBot`, `MCTSBot` (Information-Set MCTS) |
| `features.py` | Observation → 346-float vector, plus a 66-action index space |
| `train.py` | AlphaZero loop: self-play on all CPU cores → train policy/value net (GPU if available) → evaluate |
| `netbot.py` | Net-guided search bot. Use it as `net[:models/best.npz][:SIMS]` in any bot spec |
| `dashboard.py` | Live terminal dashboard for a training run (`--once` for a snapshot) |
| `models/best.npz` | The trained policy/value net (3.4 MB) |
| `selfplay.py` | Standalone self-play data dump (`.npz`) for your own training experiments |
| `test_rules.py` | Rule tests |

## House rules

- Standard 52-card deck. **Aces = Wizards** (always win). **2s = Jesters** (always lose). 3–K are normal cards.
- A game is **10 rounds**. Round *r* deals *r* cards. The next card in the deck sets trump. If it's a Wizard or Jester, there's no trump.
- **Screw the dealer:** total bids can't equal the number of cards dealt. The dealer bids last and is forced to break the tie.
- You must follow the suit that was led if you can. You can always play a Wizard or a Jester.
- **Who wins a trick:** the first Wizard. Otherwise the highest trump. Otherwise the highest card of the suit that was led. If every card is a Jester, the **last** Jester wins. If a Jester leads, the first normal card played sets the suit.
- **Scoring:** hit your bid exactly and you get 20 + 10 per trick. Miss it and you lose 10 per trick you were over or under.
- *Not in the spec, so we picked:* if a Wizard leads, there's no suit to follow. The dealer moves one seat left each round.

## Why not just copy AlphaZero?

AlphaZero assumes perfect information and two players. Wizard has **hidden hands**, **4 players** and **random deals**. The standard fix:

1. **Determinization.** Guess the hidden cards in a way that matches what you've seen. For example, if someone didn't follow ♥, they have no ♥. Then search as if the game were open. `determinize()` does this.
2. **ISMCTS.** Instead of building a separate tree for each guess, share one tree across many guesses (Cowling et al., 2012). That's `bots.ismcts()`.
3. **Add learning (the AlphaZero part).** A net with a policy head (prior over moves) and a value head (expected round score) replaces random rollouts. `ismcts(value_fn=...)` is where the value net plugs in.
4. **Loop:** self-play with search → train the net on visit counts and outcomes → search gets stronger → repeat.

## Benchmarks to beat

All tables are 4 players over 10-round games, with seats rotated.

| matchup | result |
|---|---|
| `mcts:200` vs 3× `heuristic` | MCTS averages ~189, heuristics ~98 |
| `net:models/best.npz` (NetBot-200) vs `mcts:200` + 2× `heuristic` | NetBot **198.0** vs MCTS **184.3** (+13.8 over 96 games) |

The trained model in `models/best.npz` came from a run that was stopped early. The bootstrap used 4096 games of plain ISMCTS self-play; after that, 2 of 10 net-guided iterations finished (about 1 hour on an i9-13900HX plus an RTX 4080 Laptop). Each eval line has about ±15 points of noise, so read "NetBot ≈ MCTS-200, maybe slightly better" rather than a decisive win. Try it:

```bash
python tournament.py net:models/best.npz mcts:200 heuristic heuristic --deals 16
python play.py human net:models/best.npz mcts:200 heuristic      # play against it
```

### Training it yourself

```bash
python -m venv .venv && .venv/bin/pip install numpy torch
.venv/bin/python train.py            # bootstrap + 10 iterations (~5 h on 32 threads; mostly CPU self-play)
.venv/bin/python train.py --resume   # continue after stopping (reuses models/bootstrap.npz + latest.npz)
.venv/bin/python train.py --quick    # 5-minute smoke test
python dashboard.py                  # live TUI: progress, ETA, NetBot-vs-MCTS chart, CPU/GPU
```

## Ideas for your bot

- Better bidding. It's half the game, and random rollouts handle it badly.
- Smarter determinization. Weight guessed hands by how plausible each player's bid is.
- Neural policy and value nets trained from `selfplay.py` data (PyTorch).
- Pure RL (PPO or DQN) with no search at all. Does it beat MCTS?
- Opponent modeling: do the other bots overbid?
