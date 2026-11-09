"""Example player: bids its Wizards + Kings, then always tries to win when it still needs tricks."""
from engine import Bot, is_wizard, rank, trick_winner


class GreedyBot(Bot):
    name = "example_greedy"

    def act(self, obs):
        if obs.phase == "bid":
            guess = sum(is_wizard(c) or rank(c) == 13 for c in obs.hand)
            return min(obs.legal_actions, key=lambda b: abs(b - guess))
        me = obs.player
        need_more = obs.tricks_won[me] < obs.bids[me]
        winning = [c for c in obs.legal_actions if trick_winner(obs.trick + [(me, c)], obs.trump, obs.rules) == me]
        if need_more and winning:
            return winning[0]
        losing = [c for c in obs.legal_actions if c not in winning]
        return (losing or obs.legal_actions)[0]
