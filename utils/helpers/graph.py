import torch
from torch_geometric.data import Data
from copy import deepcopy
from .state import State
from .database import Database


class Graph:
    """ Game board representation and logic for GNN input. """

    _instance: 'Graph' = None

    num_nodes: int = 199
    num_taxi_edges: int = 345
    num_bus_edges: int = 99
    num_metro_edges: int = 20
    num_ferry_edges: int = 3
    graph_data_path: str = 'utils/helpers/graph_data.csv'

    # Node features for Mr. X:
    # 0 : Is Mr X truly on this node? (0 or 1)
    # 1 : Is this node reachable with a taxi ticket? (0 or 1)
    # 2 : Is this node reachable with a bus ticket? (0 or 1)
    # 3 : Is this node reachable with a metro ticket? (0 or 1)
    # 4 : Is this node reachable with a black ticket? (0 or 1)
    # 5 : On Mr X's node, can he use a double ticket at the moment? (0 or 1)
    #
    # 6 : Do detectives think Mr X might be here? (0 or 1)
    # 7 : Is there a Detective on this node? (0 or 1)
    # 8 : Could a Detective get to this node in the next move? (0 or 1)
    # 9 : The number of taxi tickets the Detective on this node has. (0 to 11 normalized)
    # 10: The number of bus tickets the Detective on this node has. (0 to 8 normalized)
    # 11: The number of metro tickets the Detective on this node has. (0 to 4 normalized)
    #
    # 12: Round number. (1 to 24 normalized)
    # 13: Number of rounds until Mr. X's next reveal. (0 to 5 normalized)
    feature_shape_x: list[list[int | float]] = [[0] * 14] * num_nodes

    # Node features for Detective:
    # 0 : Is the Detective on this node? (0 or 1)
    # 1 : Is there a different Detective on this node? (0 or 1)
    # 2 : Could Mr X be on this node? (0 or 1)
    #
    # 3 : Is this node reachable with a taxi ticket? (0 or 1)
    # 4 : Is this node reachable with a bus ticket? (0 or 1)
    # 5 : Is this node reachable with a metro ticket? (0 or 1)
    # 6 : Could another Detective get to this node in the next move? (0 or 1)
    #
    # 7 : The number of taxi tickets the Detective on this node has. (0 to 11 normalized)
    # 8 : The number of bus tickets the Detective on this node has. (0 to 8 normalized)
    # 9 : The number of metro tickets the Detective on this node has. (0 to 4 normalized)
    #
    # 10: Round number. (1 to 24 normalized)
    # 11: Number of rounds until Mr. X's next reveal. (0 to 5 normalized)
    feature_shape_d: list[list[int | float]] = [[0] * 12] * num_nodes

    def __new__(cls) -> 'Graph':
        """ Create and initialize a graph instance. """

        if cls._instance:
            return cls._instance

        cls._instance = super().__new__(cls)

        start_nodes = []
        end_nodes = []
        route_types = []
        route_encoder = {
            'taxi' : [1, 0, 0, 0],
            'bus' : [0, 1, 0, 0],
            'metro' : [0, 0, 1, 0],
            'ferry' : [0, 0, 0, 1]
        }

        cls._instance._db_rows = []
        with open(cls._instance.graph_data_path, 'r') as file:
            for line in file.read().split('\n'):
                if not line or line.startswith('#'):
                    continue

                n1, n2, route = line.split(',')
                n1, n2 = int(n1), int(n2)

                start_nodes.append(n1)
                end_nodes.append(n2)

                start_nodes.append(n2)
                end_nodes.append(n1)

                route_types.append(route_encoder[route])
                route_types.append(route_encoder[route])

                cls._instance._db_rows.append({'a' : n1, 'b' : n2, 'type' : route.strip()})

        cls._instance.edge_index = torch.tensor([start_nodes, end_nodes], dtype = torch.long)
        cls._instance.edge_attr = torch.tensor(route_types, dtype = torch.float)

        cls._instance.db = Database()
        if not cls._instance.is_db_loaded():
            cls._instance.flush_db()
            cls._instance.fill_db()

        cls._instance._potential_x_pos = []

        return cls._instance


    def is_db_loaded(self) -> bool:
        """ Check whether the graph is loaded in the database. """

        query = "MATCH (s:Station) RETURN count(s) AS station_count;"
        num_stations = self.db.query(query)['station_count']
        if num_stations != self.num_nodes:
            return False

        query = "MATCH ()-[r:CONNECTED]->() WHERE r.type = 'taxi' RETURN count(r) AS taxi_edges;"
        num_taxi_lines = self.db.query(query)['taxi_edges']
        if num_taxi_lines != self.num_taxi_edges * 2:
            return False

        query = "MATCH ()-[r:CONNECTED]->() WHERE r.type = 'bus' RETURN count(r) AS bus_edges;"
        num_bus_lines = self.db.query(query)['bus_edges']
        if num_bus_lines != self.num_bus_edges * 2:
            return False

        query = "MATCH ()-[r:CONNECTED]->() WHERE r.type = 'metro' RETURN count(r) AS metro_edges;"
        num_metro_lines = self.db.query(query)['metro_edges']
        if num_metro_lines != self.num_metro_edges * 2:
            return False

        query = "MATCH ()-[r:CONNECTED]->() WHERE r.type = 'ferry' RETURN count(r) AS ferry_edges;"
        num_ferry_lines = self.db.query(query)['ferry_edges']
        if num_ferry_lines != self.num_ferry_edges * 2:
            return False

        return True


    def fill_db(self) -> None:
        """ Load the graph in the database. """

        query = "UNWIND $rows AS row " \
                "MERGE (a:Station {id: row.a}) " \
                "MERGE (b:Station {id: row.b}) " \
                "MERGE (a)-[:CONNECTED {type: row.type}]->(b) " \
                "MERGE (b)-[:CONNECTED {type: row.type}]->(a)"
        self.db.query(query, params = {'rows' : self._db_rows})


    def flush_db(self) -> None:
        """ Flush the database. """

        query = "MATCH (n) DETACH DELETE n;"
        self.db.query(query)


    def _get_features_for_x(self, state: State) -> list[list[int | float]]:
        """ Helper function to get Mr. X's GNN input features. """

        detectives_pos = {player : pos for player, pos in state.positions.items() if player != 'x'}
        x_node = state.positions['x']
        features = deepcopy(self.feature_shape_x)

        # 0 : Is Mr X truly on this node? (0 or 1)
        features[x_node - 1][0] = 1

        # 1 : Is this node reachable with a taxi ticket? (0 or 1)
        for node in self.get_reachable_neighbors_by_ticket(state, 'x', 'taxi'):
            features[node - 1][1] = 1

        # 2 : Is this node reachable with a bus ticket? (0 or 1)
        for node in self.get_reachable_neighbors_by_ticket(state, 'x', 'bus'):
            features[node - 1][2] = 1

        # 3 : Is this node reachable with a metro ticket? (0 or 1)
        for node in self.get_reachable_neighbors_by_ticket(state, 'x', 'metro'):
            features[node - 1][3] = 1

        # 4 : Is this node reachable with a black ticket? (0 or 1)
        for node in self.get_reachable_neighbors_by_ticket(state, 'x', 'black'):
            features[node - 1][4] = 1

        # 5 : On Mr X's node, can he use a double ticket at the moment? (0 or 1)
        features[x_node - 1][5] = int(state.can_use_double)

        # 6 : Do detectives think Mr X might be here? (0 or 1)
        for node in self.get_potential_x_pos(...):
            features[node - 1][6] = 1

        # 7 : Is there a Detective on this node? (0 or 1)
        for node in detectives_pos.values():
            features[node - 1][7] = 1

        # 8 : Could a Detective get to this node in the next move? (0 or 1)
        for det in detectives_pos.keys():
            for node in self.get_reachable_neighbors(state, det):
                features[node - 1][8] = 1

        # 9 : The number of taxi tickets the Detective on this node has. (0 to 11 normalized)
        # 10: The number of bus tickets the Detective on this node has. (0 to 8 normalized)
        # 11: The number of metro tickets the Detective on this node has. (0 to 4 normalized)
        for det, node in detectives_pos.items():
            features[node - 1][9] = state.tickets[det]['taxi'] / 11
            features[node - 1][10] = state.tickets[det]['bus'] / 8
            features[node - 1][11] = state.tickets[det]['metro'] / 4

        # 12: Round number. (1 to 24 normalized)
        # 13: Number of rounds until Mr. X's next reveal. (0 to 5 normalized)
        rounds_until_reveal = state.get_steps_until_reveal()
        for node in range(self.num_nodes):
            features[node][12] = state.x_step_count / 24
            features[node][13] = rounds_until_reveal / 5

        return features


    def _get_features_for_d(self, state: State, detective: str) -> list[list[int | float]]:
        """ Helper function to get a Detective's GNN input features. """

        other_detectives_pos = {player: pos for player, pos in state.positions.items() if player != detective}
        current_node = state.positions[detective]
        features = deepcopy(self.feature_shape_d)

        # 0 : Is the Detective on this node? (0 or 1)
        features[current_node - 1][0] = 1

        # 1 : Is there a different Detective on this node? (0 or 1)
        for node in other_detectives_pos.values():
            features[node - 1][1] = 1

        # 2 : Could Mr X be on this node? (0 or 1)
        for node in self.get_potential_x_pos(...):
            features[node - 1][2] = 1

        # 3 : Is this node reachable with a taxi ticket? (0 or 1)
        for node in self.get_reachable_neighbors_by_ticket(state, detective, 'taxi'):
            features[node - 1][3] = 1

        # 4 : Is this node reachable with a bus ticket? (0 or 1)
        for node in self.get_reachable_neighbors_by_ticket(state, detective, 'bus'):
            features[node - 1][4] = 1

        # 5 : Is this node reachable with a metro ticket? (0 or 1)
        for node in self.get_reachable_neighbors_by_ticket(state, detective, 'metro'):
            features[node - 1][5] = 1

        # 6 : Could another Detective get to this node in the next move? (0 or 1)
        for det in other_detectives_pos.keys():
            for node in self.get_reachable_neighbors(state, det):
                features[node - 1][6] = 1

        # 7 : The number of taxi tickets the Detective on this node has. (0 to 11 normalized)
        # 8 : The number of bus tickets the Detective on this node has. (0 to 8 normalized)
        # 9 : The number of metro tickets the Detective on this node has. (0 to 4 normalized)
        features[current_node - 1][7] = state.tickets[detective]['taxi'] / 11
        features[current_node - 1][8] = state.tickets[detective]['bus'] / 8
        features[current_node - 1][9] = state.tickets[detective]['metro'] / 4
        for det, node in other_detectives_pos.items():
            features[node - 1][7] = state.tickets[det]['taxi'] / 11
            features[node - 1][8] = state.tickets[det]['bus'] / 8
            features[node - 1][9] = state.tickets[det]['metro'] / 4

        # 10: Round number. (1 to 24 normalized)
        # 11: Number of rounds until Mr. X's next reveal. (0 to 5 normalized)
        rounds_until_reveal = state.get_steps_until_reveal()
        for node in range(self.num_nodes):
            features[node][12] = state.x_step_count / 24
            features[node][13] = rounds_until_reveal / 5

        return features


    def get_features_by_player(self, state: State, player: str) -> torch.Tensor:
        """
        Get a player's respective GNN input features.

        Arguments:
            state: The current game state.
            player: The player name ('x', 'r', 'g', 'b', 'o', 'p').

        Returns:
            The GNN input features as a torch tensor.
        """

        features = self._get_features_for_x(state) if player == 'x' else self._get_features_for_d(state, player)
        return torch.tensor(features, dtype = torch.float)


    def get_gnn_input(self, state: State, player: str) -> Data:
        """
        Get a player's respective GNN input data.

        Arguments:
            state: The current game state.
            player: The player name ('x', 'r', 'g', 'b', 'o', 'p').

        Returns:
            The GNN input data object.
        """

        return Data(
            x = self.get_features_by_player(state, player),
            edge_index = self.edge_index,
            edge_attr = self.edge_attr
        )


    def get_reachable_neighbors_by_ticket(self, state: State, player: str, ticket: str,
                                          from_node: int = None) -> list[int]:
        """
        Get reachable neighboring stations based on the player's remaining tickets of the specified type.

        Arguments:
            state: The current game state.
            player: The player name ('x', 'r', 'g', 'b', 'o', 'p').
            ticket: The type of ticket ('taxi', 'bus', 'metro', 'black').
            from_node: The node to use as the player's current position.
                       If not specified, the player's real position will be used.

        Returns:
            The reachable neighboring stations.
        """

        if state.tickets[player][ticket] == 0:
            return []

        player_station = state.positions[player] if not from_node else from_node

        query = "MATCH (s:Station {id: $player_station})-[:CONNECTED {type: $ticket_type}]->(neighbor:Station) " \
                "RETURN DISTINCT neighbor.id AS node"
        if ticket == 'black':
            query = "MATCH (s:Station {id: $player_station})-->(neighbor:Station) RETURN DISTINCT neighbor.id AS node"

        result = self.db.query(query, params = {
            'player_station' : player_station, 'ticket_type' : ticket
        })

        if isinstance(result, dict):
            return [result['node']]

        return [record['node'] for record in result]


    def get_reachable_neighbors(self, state: State, player: str) -> list[int]:
        """
        Get reachable neighboring stations based on the player's remaining tickets.

        Arguments:
            state: The current game state.
            player: The player name ('x', 'r', 'g', 'b', 'o', 'p').

        Returns:
            The reachable neighboring stations.
        """

        nodes = set()
        for ticket, remaining in state.tickets[player].items():
            if remaining != 0 and ticket != 'double':
                nodes.update(self.get_reachable_neighbors_by_ticket(state, player, ticket))

        return list(nodes)


    def get_potential_x_pos(self, state: State) -> list[int]:
        """
        Get Mr. X's potential positions based on his travel logs.

        Arguments:
            state: The current game state.

        Returns:
            The nodes that could be occupied by Mr. X.
        """

        if len(state.travel_logs) == 0:
            self._potential_x_pos = state.X_STARTER_NODES
            return self._potential_x_pos

        last_action = state.travel_logs[-1]
        if isinstance(last_action, int):
            self._potential_x_pos = [last_action]
            return self._potential_x_pos

        new_potential_x_pos = set()
        for node in self._potential_x_pos:
            new_potential_x_pos.update(
                self.get_reachable_neighbors_by_ticket(state, 'x', ticket = last_action, from_node = node)
            )

        self._potential_x_pos = list(new_potential_x_pos)
        return self._potential_x_pos


__all__ = ["Graph"]