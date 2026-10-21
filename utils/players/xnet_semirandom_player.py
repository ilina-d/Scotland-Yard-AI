from .xnet_player import XNetPlayer
from utils.helpers import State
from utils.neural_nets.mrx_nets import BaseXNet
import random


class XNetSemiRandomPlayer(XNetPlayer):
    """ Player agent that follows a given policy with occasional random moves. """

    def __init__(self, net: BaseXNet, rand_freq: float) -> None:
        """
        Create a player instance.

        Arguments:
            net: A pretrained Mr. X GNN.
            rand_freq: How often to make a random move.
        """

        super().__init__(net = net)
        self.rand_freq = rand_freq


    def make_move(self, state: State) -> tuple[str, int | None]:
        if random.random() <= self.rand_freq:
            legal_moves = self.get_legal_moves(state)

            if legal_moves:
                return random.choice(legal_moves)

            return '', None

        return super().make_move(state)


__all__ = ['XNetSemiRandomPlayer']
