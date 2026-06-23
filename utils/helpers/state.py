import random

from utils.helpers.graph import Graph


class State:
    """ Representation of the game state. """

    positions: dict[str, int] = {'x': None, 'r': None, 'g': None, 'b': None, 'o': None, 'p': None}
    tickets: dict[str, dict[str, int]] = {'x': None, 'r': None, 'g': None, 'b': None, 'o': None, 'p': None}
    travel_logs: list[str | int] = []

    X_STARTER_NODES = (35, 45, 51, 71, 78, 104, 106, 127, 132, 146, 166, 170, 172)
    D_STARTER_NODES = (13, 26, 29, 34, 50, 53, 91, 94, 103, 112, 117, 123, 138, 141, 155, 174)
    X_STARTER_TICKETS = {'taxi': 1000, 'bus': 1000, 'metro': 1000, 'black': 5, "double": 2}
    D_STARTER_TICKETS = {'taxi': 11, 'bus': 8, 'metro': 4, 'black': 0, "double": 0}
    X_REVEAL_STEPS = (3, 8, 13, 18, 24)


    def __init__(self, graph: Graph):
        """ Create and initialize a game state instance. """

        self.graph = graph
        self.reset_game_state()


    def reset_game_state(self) -> None:
        """ Reset the game state to a starting configuration. """

        self.positions['x'] = random.choice(self.X_STARTER_NODES)
        self.tickets['x'] = self.X_STARTER_TICKETS.copy()

        self.positions['r'], self.positions['g'], self.positions['b'], self.positions['o'], self.positions['p'] = \
            random.sample(self.D_STARTER_NODES, k = 5)
        self.tickets['r'] = self.D_STARTER_TICKETS.copy()
        self.tickets['g'] = self.D_STARTER_TICKETS.copy()
        self.tickets['b'] = self.D_STARTER_TICKETS.copy()
        self.tickets['o'] = self.D_STARTER_TICKETS.copy()
        self.tickets['p'] = self.D_STARTER_TICKETS.copy()


        self.travel_logs = []


    def update_use_double_ticket(self) -> None:
        """ Update the game state after Mr. X uses a double move ticket. """

        self.tickets['x']['double'] -= 1


    def update_after_move(self, player: str, destination_node: int, ticket_used: str) -> None:
        """
        Updates the game state after a player moves to a different node on the board.

        Arguments:
            player: The player that made a move on the board.
            destination_node: The node the player moved to.
            ticket_used: The type of transport the player used.
        """

        self.positions[player] = destination_node
        self.tickets[player][ticket_used] -= 1

        if player != 'x':
            return

        if len(self.travel_logs) + 1 in self.X_REVEAL_STEPS:
            self.travel_logs.append(destination_node)
        else:
            self.travel_logs.append(ticket_used)


    def check_caught(self, detective_name: str) -> bool:
        """
        Check whether Mr. X has been caught by the specified detective.

        Arguments:
            detective_name: The name of the detective ("r", "g", "b", "o", or "p").

        Returns:
            Whether Mr. X is caught.
        """

        return self.positions['x'] == self.positions[detective_name.lower()]


    def check_cornered(self) -> bool:
        """
        Check whether Mr. X has been cornered by the detectives.

        Returns:
             Whether Mr. X is cornered.
        """

        detectives_pos = (
            self.positions['r'], self.positions['g'], self.positions['b'], self.positions['o'], self.positions['p']
        )

        x_pos = self.positions['x']

        nodes = self.graph.get_neighbors(x_pos)
        if self.tickets['x']['black'] == 0:
            ferry_nodes = self.graph.get_neighbors_by_route(x_pos, 'ferry')
            nodes.difference_update(ferry_nodes)

        for node in nodes:
            if node not in detectives_pos:
                return False

        return True


__all__ = ["State"]
