"""Template for your own bot. Copy to players/<yourname>.py and go wild.

Run it:  python challenge.py players/<yourname>.py
"""
from engine import Bot, Observation, card_str, determinize, is_wizard, is_jester, rank, suit, trick_winner


class MyBot(Bot):
    name = "my_bot"

    def act(self, obs: Observation) -> int:
        if obs.phase == "bid":
            # obs.legal_actions = allowed bids (screw-the-dealer already applied)
            wizards = sum(is_wizard(c) for c in obs.hand)
            return min(obs.legal_actions, key=lambda b: abs(b - wizards))
        # obs.legal_actions = cards you're allowed to play (follow-suit already applied)
        # Useful: obs.trick, obs.trump, obs.bids, obs.tricks_won, obs.voids, obs.unseen_cards()
        # determinize(obs) samples a full game state you can simulate on (state.copy/apply/legal_actions)
        return obs.legal_actions[0]
