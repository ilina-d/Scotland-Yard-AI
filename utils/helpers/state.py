import random

from utils.helpers.graph import Graph


class State:
    """ Representation of the game state. """

    positions: dict[str, int] = {'x': None, 'r': None, 'g': None, 'b': None, 'o': None, 'p': None}
    tickets: dict[str, dict[str, int]] = {'x': None, 'r': None, 'g': None, 'b': None, 'o': None, 'p': None}

    X_STARTER_NODES = (35, 45, 51, 71, 78, 104, 106, 127, 132, 146, 166, 170, 172)
    D_STARTER_NODES = (13, 26, 29, 34, 50, 53, 91, 94, 103, 112, 117, 123, 138, 141, 155, 174)


    def __init__(self, graph: Graph):
        """ Create and initialize a game state instance. """

        self.graph = graph
        self.reset_game_state()


    def reset_game_state(self) -> None:
        """ Reset the game state to a starting configuration. """

        self.positions['x'] = random.choice(self.X_STARTER_NODES)
        self.tickets['x'] = {'taxi': 1000, 'bus': 1000, 'metro': 1000, 'black': 5, "double": 2}

        self.positions['r'], self.positions['g'], self.positions['b'], self.positions['o'], self.positions['p'] = \
            random.sample(self.D_STARTER_NODES, k = 5)
        self.tickets['r'], self.tickets['g'], self.tickets['b'], self.tickets['o'], self.tickets['p'] = \
            [{'taxi': 11, 'bus': 8, 'metro': 4, 'black': 0, "double": 0}] * 5


    def update_state(self, player: str, destination_node: int, ticket_used: str) -> None:
        """
        Updates the game state after a player moves to a different node on the board.

        Arguments:
            player: The player that made a move on the board.
            destination_node: The node the player moved to.
            ticket_used: The type of transport the player used.
        """

        self.positions[player] = destination_node
        self.tickets[player][ticket_used] -= 1


__all__ = ["State"]
