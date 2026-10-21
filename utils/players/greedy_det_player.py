from .base_player import BasePlayer
from utils.helpers import State
import random


class GreedyDetPlayer(BasePlayer):
    """ Player agent that moves towards potential Mr. X positions when possible. """

    def make_move(self, state: State) -> tuple[str, int | None]:
        legal_moves = self.get_legal_moves(state)
        if not legal_moves:
            return '', None

        destinations = self.graph.get_potential_x_pos(state)
        for ticket, node in legal_moves:
            if node in destinations:
                return ticket, node

        return random.choice(legal_moves)


__all__ = ['GreedyDetPlayer']
