from abc import ABC, abstractmethod
from utils.helpers import State


class BasePlayer(ABC):
    """ Abstract representation of a player. """

    def __init__(self, name: str) -> None:
        """
        Create a player instance.

        Arguments:
            name: Name of the player ("x", "r", "g", "b", "o", or "p").
        """

        self.name = name


    def get_legal_moves(self, state: State, can_use_double: bool) -> list[tuple[str, int | None]]:
        """
        Get the player's legal moves for the given state.

        Arguments:
             state: The current game state.
             can_use_double: Whether the player is allowed to use a double move ticket.

        Returns:
            A list of ticket-destination tuples for all legal moves.
            Destination is None if ticket is "double".
        """

        legal_moves = []
        position, tickets = state.positions[self.name], state.tickets[self.name]
        detectives_pos = (
            state.positions['r'], state.positions['g'], state.positions['b'], state.positions['o'], state.positions['p']
        )

        for ticket, count in tickets.items():
            if count == 0 or ticket in ('double', 'black'):
                continue

            for node in state.graph.get_neighbors_by_route(position, ticket):
                if node not in detectives_pos:
                    legal_moves.append((ticket, node))

        if can_use_double and tickets['double'] != 0:
            legal_moves.append(('double', None))

        if tickets['black'] != 0:
            for node in state.graph.get_neighbors(position):
                if node not in detectives_pos:
                    legal_moves.append(('black', node))

        return legal_moves


    @abstractmethod
    def make_move(self, state: State, can_use_double: bool = True) -> tuple[str, int | None]:
        """
        Make a move on the game board.

        Arguments:
             state: The current game state.
             can_use_double: Whether the player is allowed to use a double move ticket.

        Returns:
            The chosen move in ticket-destination format.
            Destination is None if ticket is "double" or if there aren't any legal moves.
        """

        pass


__all__ = ["BasePlayer"]
