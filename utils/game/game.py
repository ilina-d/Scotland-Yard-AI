from utils.players import BasePlayer, UserPlayer
from utils.helpers import State, Graph

class Game:
    """ Scotland Yard game logic and flow. """

    def __init__(self, detective_r: BasePlayer, detective_g: BasePlayer, detective_b: BasePlayer,
                 detective_o: BasePlayer, detective_p: BasePlayer, mr_x: BasePlayer, visuals: bool = True,
                 wait_after_move: int | str | None = 'input', graph: Graph = None) -> None:
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
            graph: The game board represented as a Graph object.
                   Defaults to standard Scotland Yard board.
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
        self.wait_after_move = wait_after_move
        # Needs to be handled by pygame ^

        if graph is None:
            graph = Graph('graphs/original.csv', first_node = 1, last_node = 199)
        self.graph = graph

        self.state = State(graph)


    def reset_state(self) -> None:
        """ Reset the game state to its starting form. """

        self.state.reset_game_state()


    def check_if_caught(self):
        pass


    def make_move(self, player):
        pass


    def play(self) -> None:
        """ A """

__all__ = ["Game"]