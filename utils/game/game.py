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


    def play(self) -> str:
        """
        Start a new game.

        Returns:
              The team who won, either "Mr. X" or "Detectives".
        """

        self.state.reset_game_state()
        round_count = 0
        winner = None

        print(f'-------------------')
        for s in 'xrgbop':
            print(
                f'{s.upper()}: {self.state.positions[s]} | T: {self.state.tickets[s]["taxi"]}, B: {self.state.tickets[s]["bus"]}, M: {self.state.tickets[s]["metro"]}, F: {self.state.tickets[s]["black"]}, D: {self.state.tickets[s]["double"]}')
        print(f'-------------------')

        while winner is None:
            round_count += 1
            print(f'\n\n==[ Round {round_count} ]==')

            x_ticket, x_node = self.mr_x.make_move(self.state)
            if x_ticket == 'double':
                self.state.update_use_double_ticket()
                print(f'> Mr. X used double-move ticket.')

                x_ticket, x_node = self.mr_x.make_move(self.state, can_use_double = False)
                print(f'~-> Mr. X used {x_ticket} to move from {self.state.positions["x"]} to {x_node}.')
                self.state.update_after_move('x', x_node, x_ticket)
                x_ticket, x_node = self.mr_x.make_move(self.state, can_use_double = False)
                print(f'~-> Mr. X used {x_ticket} to move from {self.state.positions["x"]} to {x_node}.')
                self.state.update_after_move('x', x_node, x_ticket)

            else:
                print(f'> Mr. X used {x_ticket} to move from {self.state.positions["x"]} to {x_node}.')
                self.state.update_after_move('x', x_node, x_ticket)

            d_move_count = 0
            for detective in self.detectives:
                d_ticket, d_node = detective.make_move(self.state)
                if d_node is None:
                    print(f'> Detective {detective.name.upper()} has no legal moves.')
                    continue

                d_move_count += 1
                print(f'> Detective {detective.name.upper()} used {d_ticket} to move from {self.state.positions[detective.name]} to {d_node}.')
                self.state.update_after_move(detective.name, d_node, d_ticket)

                if self.state.check_caught(detective.name):
                    print(f'~-> Detectives caught Mr. X!')
                    winner = 'Detectives'
                    break

            if winner:
                break

            if self.state.check_cornered():
                print(f'~-> Detectives cornered Mr. X!')
                winner = 'Detectives'
                break

            if round_count == 22 or d_move_count == 0:
                print(f'~-> Mr. X evaded the Detectives!')
                winner = 'Mr. X'
                break

        return winner


__all__ = ["Game"]
