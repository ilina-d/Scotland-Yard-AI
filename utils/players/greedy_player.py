from .base_player import BasePlayer
from utils.helpers import State
import random


class GreedyPlayer(BasePlayer):
    """
    Player agent that makes greedy moves.
    As a detective, moves towards potential Mr. X positions.
    As Mr. X, moves away from nearby detectives.
    """

    def _make_move_x(self, state: State, legal_moves: list[tuple[str, int | None]]) -> tuple[str, int | None]:
        """ Helper function for making moves as Mr. X. """

        mrx_stations = self.graph.get_potential_x_pos(state)
        danger_nodes = set()
        for det, node in state.positions.items():
            if det == 'x':
                continue

            for neighbor in self.graph.get_reachable_neighbors(state, det):
                if neighbor in mrx_stations:
                    danger_nodes.add(neighbor)

        safe_moves = []
        for ticket, node in legal_moves:
            if node in danger_nodes:
                continue
            safe_moves.append((ticket, node))

        if safe_moves:
            return random.choice(safe_moves)

        return random.choice(legal_moves)


    def _make_move_d(self, state: State, legal_moves: list[tuple[str, int | None]]) -> tuple[str, int | None]:
        """ Helper function for making moves as a Detective. """

        mrx_stations = self.graph.get_potential_x_pos(state)
        best_ticket, best_node, best_score = 'taxi', None, -1
        for ticket, node in legal_moves:
            if node in mrx_stations:
                return ticket, node

            score = 0
            for neighbor in self.graph.get_reachable_neighbors(state, self.name, from_node = node):
                if neighbor in mrx_stations:
                    score += 1

            if score >= best_score:
                best_ticket, best_node = ticket, node

        return best_ticket, best_node


    def make_move(self, state: State) -> tuple[str, int | None]:
        legal_moves = self.get_legal_moves(state)
        if not legal_moves:
            return '', None

        if self.name == 'x':
            return self._make_move_x(state, legal_moves)

        return self._make_move_d(state, legal_moves)


__all__ = ['GreedyPlayer']
