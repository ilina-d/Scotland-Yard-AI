from torch import nn
from torch.nn import ModuleList, ReLU, Linear, Tanh
from torch_geometric.nn import RGCNConv
from .base_mrx_net import BaseMrXNet


class TestMrXNet(BaseMrXNet):
    """ Basic Mr. X GNN for testing. """

    def __init__(self) -> None:
        super().__init__()

        self.net_body = ModuleList([
            RGCNConv(self.IN_FEATURES, 64, self.NUM_RELATIONS),
            ReLU(),
            RGCNConv(64, 64, self.NUM_RELATIONS),
            ReLU(),
            RGCNConv(64, 64, self.NUM_RELATIONS),
            ReLU()
        ])

        self.policy_head = ModuleList([
            Linear(64, 64),
            ReLU(),
            Linear(64, self.NUM_TICKETS)
        ])

        self.double_head = ModuleList([
            Linear(64, 64),
            ReLU(),
            Linear(64, 1)
        ])

        self.value_head = ModuleList([
            Linear(3 * 64, 64),
            ReLU(),
            Linear(64, 1),
            Tanh()
        ])

        for head in (self.policy_head, self.double_head):
            nn.init.normal_(head[-1].weight, std = 0.01)
            nn.init.zeros_(head[-1].bias)

        self.to(self.device)


__all__ = ['TestMrXNet']
