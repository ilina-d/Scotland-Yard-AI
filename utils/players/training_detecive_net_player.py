from .detective_net_player import DetectiveNetPlayer
from utils.neural_nets.detective_nets import BaseDetectiveNet
from utils.helpers import State
import torch

class TrainingDetectiveNetPlayer(DetectiveNetPlayer):
    """ Player Agent that stores additional info useful for training Detective GNNs. """

    def __init__(self, name: str, net: BaseDetectiveNet):
        super().__init__(name, net)

        self.net.train()
        self.move_logs = []


    def make_move(self, state: State) -> tuple[str, int | None]:
        distribution = self.net.get_distribution_policy(state, self.name)
        value_before = self.net.get_state_evaluation(state, self.name, as_tensor = True).reshape(())

        if len(distribution) == 0:
            self.move_logs.append({
                'conf': value_before.new_zeros(()),
                'value_before': value_before,
                'entropy': value_before.new_zeros(()),
                'acted': False,
                'x_steps': state.x_step_count
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
            'acted': True,
            'x_steps': state.x_step_count
        })

        return ticket, node


__all__ = ['TrainingDetectiveNetPlayer']
