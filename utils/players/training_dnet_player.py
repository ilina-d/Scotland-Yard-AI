from .dnet_player import DNetPlayer
from utils.neural_nets.detective_nets import BaseDNet
from utils.helpers import State
import torch


class TrainingDNetPlayer(DNetPlayer):
    """ Player Agent that stores additional info useful for training Detective GNNs. """

    def __init__(self, name: str, net: BaseDNet) -> None:
        super().__init__(name, net)

        self.net.train()
        self.move_logs = []


    def make_move(self, state: State) -> tuple[str, int | None]:
        distribution, value_before = self.net.get_processed_output(state, self.name, value_as_tensor = True)
        value_before = value_before.reshape(())

        if len(distribution) == 0:
            self.move_logs.append({
                'conf': value_before.new_zeros(()),
                'value_before': value_before,
                'entropy': value_before.new_zeros(()),
                'acted': False
            })
            return '', None

        log_probs = torch.stack([item[2] for item in distribution])
        probs = log_probs.exp()

        move_idx = torch.multinomial(probs, 1).item()
        node, ticket, conf = distribution[move_idx]

        entropy = -(probs * torch.nan_to_num(log_probs, neginf = 0.0)).sum(dim = -1)

        self.move_logs.append({
            'conf': conf,
            'value_before': value_before,
            'entropy': entropy,
            'acted': True
        })

        return ticket, node


__all__ = ['TrainingDNetPlayer']
