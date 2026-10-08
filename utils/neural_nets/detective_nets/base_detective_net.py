from torch import nn
import torch
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data
from utils.helpers import Graph, State


class BaseDetectiveNet(nn.Module):
    """ Base Detective GNN. """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_TICKETS = 3
    TICKET_NAMES = ('taxi', 'bus', 'metro')

    def __init__(self) -> None:
        """ Create a new BaseDetectiveNet instance. """

        super().__init__()

        self.net = None
        self.policy_head = None
        self.value_head = None
        self.graph = Graph()


    def load_weights(self, file_name: str) -> None:
        """
        Load the weights from a pretrained network (at trained_nets/detective_nets/file_name.pt).

        Arguments:
            file_name: The file name to load from.
        """

        self.load_state_dict(torch.load(f'trained_nets/detective_nets/{file_name}.pt', map_location=self.device))


    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        """ Send an input though the net and get an output consisting of the policy and value. """

        outputs, edge_index, edge_attr, edge_type = data.x, data.edge_index, data.edge_attr, data.edge_type

        for layer in self.net:
            if isinstance(layer, RGCNConv):
                outputs = layer(outputs, edge_index, edge_type)
            elif isinstance(layer, GATConv):
                outputs = layer(outputs, edge_index, edge_attr)
            else:
                outputs = layer(outputs)

        policy_logits = outputs
        for layer in self.policy_head:
            policy_logits = layer(policy_logits)
        policy_logits = policy_logits.squeeze(-1)

        pooled = global_mean_pool(outputs, batch=None)
        value = pooled
        for layer in self.value_head:
            value = layer(value)
        value = value.squeeze(-1)

        return policy_logits, value


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

        legal_mask = torch.zeros(self.graph.num_nodes, self.NUM_TICKETS, dtype=torch.bool, device=self.device)
        detectives_pos = {state.positions[d] for d in ('r', 'g', 'b', 'o', 'p')}

        for t, ticket in enumerate(self.TICKET_NAMES):
            if state.tickets[player][ticket] == 0:
                continue
            for node in self.graph.get_reachable_neighbors_by_ticket(state, player, ticket):
                if node not in detectives_pos:
                    legal_mask[node - 1, t] = True

        legal_mask = legal_mask.flatten()
        outputs = outputs.flatten()

        outputs = outputs.masked_fill(~legal_mask, float('-inf'))
        probs = F.log_softmax(outputs, dim=0)

        actions = []
        for node in range(1, self.graph.num_nodes + 1):
            for t, ticket in enumerate(self.TICKET_NAMES):
                actions.append((node, ticket, probs[(node - 1) * self.NUM_TICKETS + t]))

        return actions


    def get_state_evaluation(self, state: State, player: str) -> float:
        """
        Get the estimated state value for the given player.

        Arguments:
            state: The current game state.
            player: Which detective is playing ('r', 'g', 'b', 'o', or 'p').

        Returns:
            The estimated value.
        """

        data = self.graph.get_gnn_input(state, player).to(self.device)
        output = self.forward(data)[1]

        return float(output.item())


    def get_action(self, state: State, player: str) -> tuple[str, int]:
        """
        Get the best action for the detective the given state.

        Arguments:
            state: The current game state.
            player: Which detective is playing ('r', 'g', 'b', 'o', or 'p').

        Returns:
            The chosen action by the net.
        """

        distribution = self.get_distribution_policy(state, player)
        best_action = max(distribution, key=lambda x: x[2].item())
        return best_action[1], best_action[0]


__all__ = ['BaseDetectiveNet']
