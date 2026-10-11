from .base_player import BasePlayer
from utils.neural_nets.detective_nets import BaseDetectiveNet
from utils.helpers import State


class DetectiveNetPlayer(BasePlayer):
    """ Player agent that plays by a GNN's policy. """

    def __init__(self, name: str, net: BaseDetectiveNet) -> None:
        """
        Create a player instance.

        Arguments:
            name: Name of the player ("r", "g", "b", "o", or "p").
            net: A pretrained detective GNN.
        """

        super().__init__(name = name)
        self.net = net
        self.net.eval()


    def make_move(self, state: State) -> tuple[str, int | None]:
        return self.net.get_action(state, self.name)


__all__ = ['DetectiveNetPlayer']
