from .base_player import BasePlayer
from utils.neural_nets.mrx_nets import BaseXNet
from utils.helpers import State


class XNetPlayer(BasePlayer):
    """ Player agent that plays by a GNN's policy. """

    def __init__(self, net: BaseXNet) -> None:
        """
        Create a player instance.

        Arguments:
            net: A pretrained Mr. X GNN.
        """

        super().__init__(name = 'x')
        self.net = net
        self.net.eval()


    def make_move(self, state: State) -> tuple[str, int | None]:
        return self.net.get_action(state)
