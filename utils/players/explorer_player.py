from .base_player import BasePlayer
from utils.helpers import State

import random


class ExplorerPlayer(BasePlayer):
    """ Player agent that prefers moving towards unexplored nodes. """

    seen_nodes: set[int] = set()

    def make_move(self, state: State) -> tuple[str, int | None]:
        legal_moves = self.get_legal_moves(state)

        if not legal_moves:
            return '', None

        current_pos = state.positions[self.name]
        self.seen_nodes.add(current_pos)

        new_legal_moves = [
            (ticket, node) for ticket, node in legal_moves
            if ticket is not None and node not in self.seen_nodes
        ]

        if not new_legal_moves:
            ticket, node = random.choice(legal_moves)
        else:
            ticket, node = random.choice(new_legal_moves)

        self.seen_nodes.add(node)
        return ticket, node

__all__ = ['ExplorerPlayer']
