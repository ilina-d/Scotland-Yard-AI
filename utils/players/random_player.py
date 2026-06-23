from .base_player import BasePlayer
from utils.helpers import State

import random


class RandomPlayer(BasePlayer):
    """ Player agent making random moves. """

    def make_move(self, state: State, can_use_double: bool = True) -> tuple[str, int | None]:
        legal_moves = self.get_legal_moves(state, can_use_double)

        if legal_moves:
            return random.choice(legal_moves)

        return '', None


__all__ = ['RandomPlayer']
