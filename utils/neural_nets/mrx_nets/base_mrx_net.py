from torch import nn
import torch
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv, GATConv
from torch_geometric.data import Data
from utils.helpers import Graph, State


class BaseMrXNet(nn.Module):
    """ Base Mr. X GNN. """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_TICKETS = 4
    TICKET_NAMES = ('taxi', 'bus', 'metro', 'black')

    net_body: nn.ModuleList
    policy_head: nn.ModuleList
    double_head: nn.ModuleList
    value_head: nn.ModuleList


    def __init__(self) -> None:
        """ Create a new BaseMrXNet instance. """

        super().__init__()
        self.graph = Graph()

        self.IN_FEATURES = len(self.graph.feature_shape_x[0])
        self.NUM_RELATIONS = self.graph.num_edge_types


    def load_weights(self, file_name: str) -> None:
        """
        Load the weights from a pretrained network (at trained_nets/mrx_nets/file_name.pt).

        Arguments:
            file_name: The file name to load from.
        """

        self.load_state_dict(torch.load(f'trained_nets/mrx_nets/{file_name}.pt', map_location = self.device))


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

        double_output = outputs[data.x[:, 0] == 1]
        for layer in self.double_head:
            double_output = layer(double_output)
        double_output = double_output.squeeze(-1)

        self_emb = outputs[data.x[:, 0].argmax()].unsqueeze(0)
        det_mean = self._masked_mean(outputs, data.x[:, 7] == 1)
        mrx_mean = self._masked_mean(outputs, data.x[:, 6] == 1)

        value_output = torch.cat([self_emb, det_mean, mrx_mean], dim = -1)
        for layer in self.value_head:
            value_output = layer(value_output)
        value_output = value_output.squeeze(-1)

        policy_output = torch.cat([policy_output.flatten(), double_output.flatten()])
        return policy_output, value_output


    def get_processed_output(self, state: State, value_as_tensor: bool = False, value_only: bool = False) \
            -> tuple[list[tuple[int, str, torch.Tensor]], float | torch.Tensor]:
        """
        Get the processed output of the network for a given input.

        Arguments:
            state: The current game state.
            value_as_tensor: Whether to return the state evaluation as a tensor.
            value_only: Whether to return early without processing the action tuples.

        Returns:
            Two things. First, a list of (destination, ticket, log_probability) tuples for each action.
            Second, the estimated evaluation of the given state as a float or tensor.
        """

        player = 'x'
        data = self.graph.get_gnn_input(state, player).to(self.device)
        policy_output, value_output = self.forward(data)

        if not value_as_tensor:
            value_output = float(value_output.item())

        if value_only:
            return [], value_output

        legal_mask = torch.zeros(self.NUM_TICKETS * self.graph.num_nodes + 1, dtype = torch.bool)
        detectives_pos = {state.positions[d] for d in ('r', 'g', 'b', 'o', 'p')}

        has_legal_moves = False
        for t, ticket in enumerate(self.TICKET_NAMES):
            if state.tickets[player][ticket] == 0:
                continue
            for node in self.graph.get_reachable_neighbors_by_ticket(state, player, ticket):
                if node not in detectives_pos and state.positions['x']:
                    legal_mask[(node - 1) * self.NUM_TICKETS + t] = True
                    has_legal_moves = True

        if has_legal_moves and state.can_use_double and state.tickets[player]['double']:
            legal_mask[-1] = True

        if not legal_mask.any():
            return [], value_output

        legal_mask = legal_mask.to(self.device)
        outputs = policy_output.masked_fill(~legal_mask, float('-inf'))
        probs = F.log_softmax(outputs, dim = 0)

        actions = []
        for node in range(1, self.graph.num_nodes + 1):
            for t, ticket in enumerate(self.TICKET_NAMES):
                actions.append((node, ticket, probs[(node - 1) * self.NUM_TICKETS + t]))

        actions.append((-1, 'double', probs[-1]))
        
        return actions, value_output


    @torch.no_grad()
    def get_action(self, state: State) -> tuple[str, int | None]:
        """
        Get the best action for the detective the given state.

        Arguments:
            state: The current game state.

        Returns:
            The chosen action by the net.
        """

        distribution, _ = self.get_processed_output(state)
        if len(distribution) == 0:
            return '', None

        probs = torch.stack([item[2] for item in distribution])

        action_idx = probs.argmax().item()
        if action_idx == probs.numel() - 1:
            return 'double', None

        node, ticket = divmod(action_idx, self.NUM_TICKETS)
        return self.TICKET_NAMES[ticket], node + 1


__all__ = ['BaseMrXNet']
