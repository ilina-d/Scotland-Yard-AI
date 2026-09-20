import random


class State:
    """ Representation of the game state. """

    positions: dict[str, int] = {'x': None, 'r': None, 'g': None, 'b': None, 'o': None, 'p': None}
    tickets: dict[str, dict[str, int]] = {'x': None, 'r': None, 'g': None, 'b': None, 'o': None, 'p': None}
    travel_logs: list[str | int] = []
    reveal_logs: dict[int, str] = {}
    can_use_double: bool = None
    x_step_count: int = None

    X_STARTER_NODES = (35, 45, 51, 71, 78, 104, 106, 127, 132, 146, 166, 170, 172)
    D_STARTER_NODES = (13, 26, 29, 34, 50, 53, 91, 94, 103, 112, 117, 123, 138, 141, 155, 174)
    X_STARTER_TICKETS = {'taxi': 1000, 'bus': 1000, 'metro': 1000, 'black': 5, "double": 2}
    D_STARTER_TICKETS = {'taxi': 11, 'bus': 8, 'metro': 4, 'black': 0, "double": 0}
    X_REVEAL_STEPS = (3, 8, 13, 18, 24)


    def __init__(self):
        """ Create and initialize a game state instance. """

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
        self.can_use_double = True
        self.x_step_count = 1


    def update_use_double_ticket(self) -> None:
        """ Update the game state after Mr. X uses a double move ticket. """

        self.tickets['x']['double'] -= 1


    def update_double_ticket_permission(self, value: bool) -> None:
        """
        Update the can_use_double parameter with the given value.

        Arguments:
            value: True or False
        """

        self.can_use_double = value


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

        self.x_step_count += 1

        if self.x_step_count in self.X_REVEAL_STEPS:
            self.travel_logs.append(destination_node)
            self.reveal_logs[self.x_step_count] = ticket_used
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


    def get_steps_until_reveal(self) -> int:
        """ Get the number of steps until Mr. X's next reveal. """

        steps = self.X_REVEAL_STEPS[self.x_step_count // 5] - self.x_step_count
        if steps == -1:
            steps = self.x_step_count // 19 + 4

        return steps


__all__ = ["State"]