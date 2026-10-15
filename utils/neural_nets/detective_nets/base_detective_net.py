from torch import nn
import torch
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv, GATConv
from torch_geometric.data import Data
from utils.helpers import Graph, State


class BaseDetectiveNet(nn.Module):
    """ Base Detective GNN. """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_TICKETS = 3
    TICKET_NAMES = ('taxi', 'bus', 'metro')

    net_body: nn.ModuleList
    policy_head: nn.ModuleList
    value_head: nn.ModuleList


    def __init__(self) -> None:
        """ Create a new BaseDetectiveNet instance. """

        super().__init__()
        self.graph = Graph()

        self.IN_FEATURES = len(self.graph.feature_shape_d[0])
        self.NUM_RELATIONS = self.graph.num_edge_types


    def load_weights(self, file_name: str) -> None:
        """
        Load the weights from a pretrained network (at trained_nets/detective_nets/file_name.pt).

        Arguments:
            file_name: The file name to load from.
        """

        self.load_state_dict(torch.load(f'trained_nets/detective_nets/{file_name}.pt', map_location = self.device))


    @staticmethod
    def _masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Helper function for computing a masked mean over the given tensor. """

        mask = mask.float().unsqueeze(-1)
        return (tensor * mask).sum(dim = 0, keepdim = True) / mask.sum().clamp(min = 1)


    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """ Process the given input through each of the network's layers. """

        outputs, edge_index, edge_attr, edge_type = data.x, data.edge_index, data.edge_attr, data.edge_type

        for layer in self.net_body:
            if isinstance(layer, RGCNConv):
                outputs = layer(outputs, edge_index, edge_type)
            elif isinstance(layer, GATConv):
                outputs = layer(outputs, edge_index, edge_attr)
            else:
                outputs = layer(outputs)

        policy_output = outputs
        for layer in self.policy_head:
            policy_output = layer(policy_output)

        self_emb = outputs[data.x[:, 0] == 1]
        det_mean = self._masked_mean(outputs, data.x[:, 1] == 1)
        mrx_mean = self._masked_mean(outputs, data.x[:, 2] == 1)

        value_output = torch.cat([self_emb, det_mean, mrx_mean], dim = -1)
        for layer in self.value_head:
            value_output = layer(value_output)
        value_output = value_output.squeeze(-1)

        return policy_output, value_output


    def get_distribution_policy(self, state: State, player: str) -> list[tuple[int, str, torch.Tensor]]:
        """
        Get the processed output of the network for a given input.

        Arguments:
            state: The current game state.
            player: Which detective is deciding ('r', 'g', 'b', 'o', or 'p').

        Returns:
            A list of tuples (destination, ticket, log_probability), for each destination node on the map.
        """

        data = self.graph.get_gnn_input(state, player).to(self.device)
        outputs = self.forward(data)[0]

        legal_mask = torch.zeros(self.graph.num_nodes, self.NUM_TICKETS, dtype = torch.bool, device = self.device)
        detectives_pos = {state.positions[d] for d in ('r', 'g', 'b', 'o', 'p')}

        for t, ticket in enumerate(self.TICKET_NAMES):
            if state.tickets[player][ticket] == 0:
                continue
            for node in self.graph.get_reachable_neighbors_by_ticket(state, player, ticket):
                if node not in detectives_pos:
                    legal_mask[node - 1, t] = True

        legal_mask = legal_mask.flatten()
        if not legal_mask.any():
            return []

        outputs = outputs.flatten()
        outputs = outputs.masked_fill(~legal_mask, float('-inf'))
        probs = F.log_softmax(outputs, dim = 0)

        actions = []
        for node in range(1, self.graph.num_nodes + 1):
            for t, ticket in enumerate(self.TICKET_NAMES):
                actions.append((node, ticket, probs[(node - 1) * self.NUM_TICKETS + t]))

        return actions


    def get_state_evaluation(self, state: State, player: str, as_tensor: bool = False) -> float | torch.Tensor:
        """
        Get the estimated state value for the given player.

        Arguments:
            state: The current game state.
            player: Which detective is playing ('r', 'g', 'b', 'o', or 'p').
            as_tensor: Whether to return the state evaluation as a tensor.

        Returns:
            The estimated value.
        """

        data = self.graph.get_gnn_input(state, player).to(self.device)
        output = self.forward(data)[1]

        if as_tensor:
            return output

        return float(output.item())


    def get_action(self, state: State, player: str) -> tuple[str, int | None]:
        """
        Get the best action for the detective the given state.

        Arguments:
            state: The current game state.
            player: Which detective is playing ('r', 'g', 'b', 'o', or 'p').

        Returns:
            The chosen action by the net.
        """

        distribution = self.get_distribution_policy(state, player)
        if len(distribution) == 0:
            return '', None

        probs = torch.stack([item[2] for item in distribution])
        node, ticket, _ = distribution[probs.argmax().item()]

        return ticket, node


__all__ = ['BaseDetectiveNet']
