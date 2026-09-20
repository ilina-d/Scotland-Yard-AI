from utils.players import BasePlayer, UserPlayer
from utils.helpers import State, Graph


class Game:
    """ Scotland Yard game logic and flow. """

    def __init__(self, detective_r: BasePlayer, detective_g: BasePlayer, detective_b: BasePlayer,
                 detective_o: BasePlayer, detective_p: BasePlayer, mr_x: BasePlayer, visuals: bool = True,
                 wait_after_move: int | str | None = 'input') -> None:
        """
        Create and initialize an instance of the Game class.

        Note on the Players:
            If a user is controlling one or more of the detectives, Mr. X cannot also be controlled by a user.

        Waiting Methods:
            - int | Number of milliseconds to wait.
            - "input" | Wait until input is given.
            - None | No waiting.

        Arguments:
            detective_r: The player that controls the red detective.
            detective_g: The player that controls the green detective.
            detective_b: The player that controls the blue detective.
            detective_o: The player that controls the orange detective.
            detective_p: The player that controls the purple detective.
            mr_x: The player that controls Mr. X.
            visuals: Whether to display game graphics.
            wait_after_move: The method for waiting after each move.
        """

        detectives = [detective_r, detective_g, detective_b, detective_o, detective_p]
        if any([isinstance(d, UserPlayer) for d in detectives]) and isinstance(mr_x, UserPlayer):
            raise Exception("A user cannot control both a detective and Mr. X in the same game.")

        self.detectives = detectives
        self.detective_r = detective_r
        self.detective_g = detective_g
        self.detective_b = detective_b
        self.detective_o = detective_o
        self.detective_p = detective_p
        self.mr_x = mr_x

        self.visuals = visuals
        self.wait_after_move = wait_after_move  # <-- needs to be handled by pygame

        self.state = State()
        self.graph = Graph()

    def play(self) -> str:
        """
        Start a new game.

        Returns:
              The team who won, either "Mr. X" or "Detectives".
        """

        self.state.reset_game_state()
        winner = None

        while winner is None:
            x_ticket, x_node = self.mr_x.make_move(self.state)
            if x_ticket == 'double':
                self.state.update_use_double_ticket()
                self.state.update_double_ticket_permission(False)

                x_ticket, x_node = self.mr_x.make_move(self.state)
                self.state.update_after_move('x', x_node, x_ticket)

                x_ticket, x_node = self.mr_x.make_move(self.state)
                self.state.update_after_move('x', x_node, x_ticket)

                self.state.update_double_ticket_permission(True)

            else:
                self.state.update_after_move('x', x_node, x_ticket)

            if self.state.x_step_count >= 24:
                winner = 'Mr. X'
                break

            d_move_count = 0
            for detective in self.detectives:
                d_ticket, d_node = detective.make_move(self.state)
                if d_node is None:
                    continue

                d_move_count += 1
                self.state.update_after_move(detective.name, d_node, d_ticket)

                if self.state.check_caught(detective.name):
                    winner = 'Detectives'
                    break

            if winner:
                break

            if d_move_count == 0:
                winner = 'Mr. X'
                break

        return winner


__all__ = ["Game"]