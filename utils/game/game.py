from utils.players import BasePlayer, UserPlayer
from utils.helpers import State, Graph
from utils.display import Display


class Game:
    """ Scotland Yard game logic and flow. """

    def __init__(self, detective_r: BasePlayer, detective_g: BasePlayer, detective_b: BasePlayer,
                 detective_o: BasePlayer, detective_p: BasePlayer, mr_x: BasePlayer, visuals: bool = True,
                 wait_after_move: int | str | None = 'input', always_show_x: bool = False) -> None:
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
            always_show_x: Whether to always show Mr. X's location.
        """

        detectives = [detective_r, detective_g, detective_b, detective_o, detective_p]
        if any([isinstance(d, UserPlayer) for d in detectives]) and isinstance(mr_x, UserPlayer):
            raise Exception("A user cannot control both a detective and Mr. X in the same game.")

        if not visuals and (any([isinstance(d, UserPlayer) for d in detectives]) or isinstance(mr_x, UserPlayer)):
            raise Exception("A user cannot play without visuals.")

        self.detectives = detectives
        self.detective_r = detective_r
        self.detective_g = detective_g
        self.detective_b = detective_b
        self.detective_o = detective_o
        self.detective_p = detective_p
        self.mr_x = mr_x

        self.visuals = visuals
        self.wait_after_move = wait_after_move
        self.always_show_x = always_show_x

        self.state = State()
        self.graph = Graph()
        self.display = Display(self.state, wait_after_move, always_show_x) if visuals else None
        self._game_phase = 'x'
        self._d_move_count = 0
        self._d_move_index = 0


    def play(self) -> str:
        """
        Start a new game.

        Returns:
              The team who won, either "Mr. X" or "Detectives".
        """

        if self.visuals:
            return self._play_with_visuals()

        return self._play_without_visuals()


    def _get_reachable_nodes(self, player: str) -> dict[int, list[str]]:
        """ Helper function for getting reachable neighboring nodes. """

        reachable_nodes = {}
        for node in self.graph.get_reachable_neighbors_by_ticket(self.state, player, 'taxi'):
            if node not in reachable_nodes:
                reachable_nodes[node] = set()
            reachable_nodes[node].add('taxi')

        for node in self.graph.get_reachable_neighbors_by_ticket(self.state, player, 'bus'):
            if node not in reachable_nodes:
                reachable_nodes[node] = set()
            reachable_nodes[node].add('bus')

        for node in self.graph.get_reachable_neighbors_by_ticket(self.state, player, 'metro'):
            if node not in reachable_nodes:
                reachable_nodes[node] = set()
            reachable_nodes[node].add('metro')

        if player == 'x':
            for node in self.graph.get_reachable_neighbors_by_ticket(self.state, player, 'black'):
                if node not in reachable_nodes:
                    reachable_nodes[node] = set()
                reachable_nodes[node].add('black')

        reachable_nodes = {node : list(tickets) for node, tickets in reachable_nodes.items()}
        return reachable_nodes


    def _run_logic_with_visuals(self, dt: float) -> None:
        """ Helper function for executing the game logic along with visuals. """

        if self.state.winner:
            return
        if self.display.paused_for_input:
            return
        if self.display.paused_time_left > 0:
            self.display.update_paused_time(dt)
            return

        if self.display.user_turn.is_user_turn and self.display.user_turn.chosen_move is None:
            return

        if self._game_phase == 'x':
            self._step_x()
        elif self._game_phase == 'd':
            self._step_d()

        self.display.start_wait_after_move()


    def _step_x(self) -> None:
        """ Helper function for running Mr. X's turn logic along with visuals. """

        self.state.set_current_player('x')

        if isinstance(self.mr_x, UserPlayer):
            if not self.display.user_turn.is_user_turn:
                self._begin_user_move('x')
                return

            if self.display.user_turn.chosen_move is None:
                return

            ticket, node = self.display.user_turn.chosen_move
            self.display.user_turn.is_user_turn = False

        else:
            ticket, node = self.mr_x.make_move(self.state)

        if ticket == 'double':
            self.state.update_use_double_ticket()
            self.state.update_double_ticket_permission(False)

            # TODO REMINDER: This would not work with a user player.
            t1, n1 = self.mr_x.make_move(self.state)
            self.state.update_after_move('x', n1, t1)

            t2, n2 = self.mr_x.make_move(self.state)
            self.state.update_after_move('x', n2, t2)

            self.state.update_double_ticket_permission(True)

        else:
            self.state.update_after_move('x', node, ticket)

        if self.state.x_step_count >= 24:
            self.state.set_winner('x')
            self._game_phase = 'over'
            return

        self._game_phase = 'd'
        self._d_move_count = 0
        self._d_move_index = 0


    def _step_d(self) -> None:
        """ Helper function for running a Detective's turn logic along with visuals. """

        if self._d_move_index == len(self.detectives):
            if self._d_move_count == 0:
                self.state.set_winner('x')
                self._game_phase = 'over'
                return

            self._game_phase = 'x'
            return

        detective = self.detectives[self._d_move_index]
        self.state.set_current_player(detective.name)

        if isinstance(detective, UserPlayer):
            if not self.display.user_turn.is_user_turn:
                self._begin_user_move(detective.name)
                return

            if self.display.user_turn.chosen_move is None:
                return

            ticket, node = self.display.user_turn.chosen_move
            self.display.user_turn.is_user_turn = False

        else:
            ticket, node = detective.make_move(self.state)

        self._d_move_index += 1
        if node is None:
            return

        self._d_move_count += 1
        self.state.update_after_move(detective.name, node, ticket)

        if self.state.check_caught(detective.name):
            self.state.set_winner(detective.name)
            return


    def _begin_user_move(self, player: str) -> None:
        """ Helper function for handling user input along with visuals. """

        reachable_nodes = self._get_reachable_nodes(player)
        if not reachable_nodes:
            if player == 'x':
                self.state.winner = 'd'
            else:
                self._d_move_index += 1
            return

        self.display.user_turn.is_user_turn = True
        self.display.user_turn.player_name = player
        self.display.user_turn.chosen_move = None
        self.display.user_turn.reachable_nodes = reachable_nodes
        self.display.user_turn.selected_node = None
        self.display.hide_popup()


    def _play_with_visuals(self) -> str:
        """ Helper function for running the game with visuals. """

        self.state.reset_game_state()
        while self.display.is_running:
            dt = self.display.clock.tick(60) / 1000
            self.display.process_events()

            if self.state.winner:
                break

            self._run_logic_with_visuals(dt)
            self.display.draw()

        return self.state.winner


    def _play_without_visuals(self) -> str:
        """ Helper function for running the game without visuals. """

        self.state.reset_game_state()
        while self.state.winner is None:
            self.state.set_current_player('x')
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
                self.state.set_winner('x')
                break

            d_move_count = 0
            for detective in self.detectives:
                self.state.set_current_player(detective.name)
                d_ticket, d_node = detective.make_move(self.state)
                if d_node is None:
                    continue

                d_move_count += 1
                self.state.update_after_move(detective.name, d_node, d_ticket)

                if self.state.check_caught(detective.name):
                    self.state.set_winner(detective.name)
                    break

            if self.state.winner:
                break

            if d_move_count == 0:
                self.state.set_winner('x')
                break

        return self.state.winner


__all__ = ["Game"]
