from torch import nn
from torch.nn import ModuleList, ReLU, Linear, Tanh
from torch_geometric.nn import RGCNConv, DeepGCNLayer
from .base_xnet import BaseXNet


class XNet5LRes(BaseXNet):
    """ 5-layer Mr. X GNN that includes residual layers. """

    def __init__(self) -> None:
        super().__init__()

        self.net_body = ModuleList([
            RGCNConv(self.IN_FEATURES, 64, self.NUM_RELATIONS),
            ReLU(),
            DeepGCNLayer(RGCNConv(64, 64, self.NUM_RELATIONS), act = ReLU(), block = 'res'),
            DeepGCNLayer(RGCNConv(64, 64, self.NUM_RELATIONS), act = ReLU(), block = 'res'),
            DeepGCNLayer(RGCNConv(64, 64, self.NUM_RELATIONS), act = ReLU(), block = 'res'),
            DeepGCNLayer(RGCNConv(64, 64, self.NUM_RELATIONS), act = ReLU(), block = 'res')
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


__all__ = ['XNet5LRes']
