from torch import nn
from torch.nn import ModuleList, ReLU, Linear, Tanh
from torch_geometric.nn import RGCNConv
from .base_xnet import BaseXNet


class XNet5LW(BaseXNet):
    """ Basic 5-layer wide Mr. X GNN. """

    def __init__(self) -> None:
        super().__init__()

        self.net_body = ModuleList([
            RGCNConv(self.IN_FEATURES, 128, self.NUM_RELATIONS),
            ReLU(),
            RGCNConv(128, 128, self.NUM_RELATIONS),
            ReLU(),
            RGCNConv(128, 128, self.NUM_RELATIONS),
            ReLU(),
            RGCNConv(128, 128, self.NUM_RELATIONS),
            ReLU(),
            RGCNConv(128, 128, self.NUM_RELATIONS),
            ReLU()
        ])

        self.policy_head = ModuleList([
            Linear(128, 128),
            ReLU(),
            Linear(128, self.NUM_TICKETS)
        ])

        self.double_head = ModuleList([
            Linear(128, 128),
            ReLU(),
            Linear(128, 1)
        ])

        self.value_head = ModuleList([
            Linear(3 * 128, 128),
            ReLU(),
            Linear(128, 1),
            Tanh()
        ])

        for head in (self.policy_head, self.double_head):
            nn.init.normal_(head[-1].weight, std = 0.01)
            nn.init.zeros_(head[-1].bias)

        self.to(self.device)


__all__ = ['XNet5LW']
