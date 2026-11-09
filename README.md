# Wizard AI Arena

Build an AI that plays **Wizard** (club house rules), then pit everyone's bots against each other.

The plan is roughly **AlphaZero for a hidden-information card game**: Monte Carlo tree search, then reinforcement learning to make the search smarter.

## Quick start (Python 3.10+, numpy)

```bash
python test_rules.py                                   # check the rules engine
python tournament.py mcts:200 heuristic heuristic random --games 4
python tournament.py mcts:100 heuristic heuristic heuristic --games 1 --rounds 3 --verbose
cp my_bot.py alice.py                                  # start your own bot
python tournament.py alice.py:MyBot bob.py:MyBot carol.py:MyBot dan.py:MyBot --games 20
```

Seats rotate, and every rotation uses the same deals, so a lucky seat or a lucky hand doesn't decide the result.

## Files

| file | what it is |
|---|---|
| `engine.py` | Rules, `RoundState` (a full state you can copy and simulate), `Observation` (what a bot sees), `determinize()`, and the `Bot` base class |
| `bots.py` | `RandomBot`, `HeuristicBot`, `MCTSBot` (Information-Set MCTS) |
| `tournament.py` | Bot-vs-bot arena. Load built-in bots or `file.py:ClassName` |
| `my_bot.py` | Template to copy |
| `features.py` | Observation → 346-float vector, plus the 66-action index space for neural nets |
| `selfplay.py` | Generates `(state, MCTS policy, outcome)` training data. This is step 1 of the AlphaZero loop |
| `test_rules.py` | Rule tests |

## House rules

- Standard 52-card deck. **Aces = Wizards** (always win). **2s = Jesters** (always lose). 3–K are normal cards.
- Round *r* deals *r* cards. With 4 players that's 13 rounds. The next card in the deck sets trump. If it's a Wizard or Jester, or no card is left, there's no trump.
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

Example run: `mcts:200` vs 3× `heuristic`, full game → MCTS averaged **255** vs about **95** and won 75% (only 4 games, so the numbers are noisy).

## Ideas for your bot

- Better bidding. It's half the game, and random rollouts handle it badly.
- Smarter determinization. Weight guessed hands by how plausible each player's bid is.
- Neural policy and value nets trained from `selfplay.py` data (PyTorch).
- Pure RL (PPO or DQN) with no search at all. Does it beat MCTS?
- Opponent modeling: do the other bots overbid?
