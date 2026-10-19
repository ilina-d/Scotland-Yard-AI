import torch
from torch_geometric.data import Data
from .state import State
from .database import Database


class Graph:
    """ Game board representation and logic for GNN input. """

    _instance: 'Graph' = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index: torch.Tensor = None
    edge_attr: torch.Tensor = None
    edge_type: torch.Tensor = None

    num_nodes: int = 199
    num_taxi_edges: int = 345
    num_bus_edges: int = 99
    num_metro_edges: int = 20
    num_ferry_edges: int = 3
    num_edge_types: int = 4
    graph_data_path: str = 'utils/helpers/graph_data.csv'

    routes: dict[str, dict[int, set[int]]] = {}
    potential_x_pos: list[int] = []

    db: Database = None
    _db_rows: list[dict[str, int | str]] = []

    # Node features for Mr. X:
    # 0 : Is Mr X truly on this node? (0 or 1)
    # 1 : Is this node reachable with a taxi ticket? (0 or 1)
    # 2 : Is this node reachable with a bus ticket? (0 or 1)
    # 3 : Is this node reachable with a metro ticket? (0 or 1)
    # 4 : Is this node reachable with a black ticket? (0 or 1)
    # 5 : On Mr X's node, can he use a double ticket at the moment? (0 or 1)
    # 6 : On Mr X's node, how many remaining black tickets does he have? (0 to 5 normalized)
    # 7 : On Mr X's node, how many remaining double tickets does he have? (0 to 2 normalized)
    #
    # 8 : Do detectives think Mr X might be here? (0 or 1)
    # 9 : Is there a Detective on this node? (0 or 1)
    # 10: Could a Detective get to this node in the next move? (0 or 1)
    # 11: The number of taxi tickets the Detective on this node has. (0 to 11 normalized)
    # 12: The number of bus tickets the Detective on this node has. (0 to 8 normalized)
    # 13: The number of metro tickets the Detective on this node has. (0 to 4 normalized)
    #
    # 14: Round number. (1 to 24 normalized)
    # 15: Number of rounds until Mr. X's next reveal. (0 to 5 normalized)
    feature_shape_x: list[list[int | float]] = [[0] * 16 for _ in range(num_nodes)]

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
    feature_shape_d: list[list[int | float]] = [[0] * 12 for _ in range(num_nodes)]


    def __new__(cls, use_neo4j: bool = False) -> 'Graph':
        """
        Create and initialize a graph instance.

        Arguments:
            use_neo4j: Whether the graph searching will go through Neo4j or python.
        """

        if cls._instance:
            return cls._instance

        cls._instance = super().__new__(cls)
        cls._instance.use_neo4j = use_neo4j

        cls._instance.routes = {'taxi': {}, 'bus': {}, 'metro': {}, 'ferry': {}}

        start_nodes, end_nodes = [], []
        route_attrs, route_types = [], []
        route_encoder = {'taxi': [1, 0, 0, 0], 'bus': [0, 1, 0, 0], 'metro': [0, 0, 1, 0], 'ferry': [0, 0, 0, 1]}

        cls._instance._db_rows = []
        with open(cls._instance.graph_data_path, 'r') as file:
            for line in file.read().split('\n'):
                if not line or line.startswith('#'):
                    continue

                n1, n2, route = line.split(',')
                n1, n2 = int(n1), int(n2)

                start_nodes.append(n1 - 1)
                end_nodes.append(n2 - 1)

                start_nodes.append(n2 - 1)
                end_nodes.append(n1 - 1)

                route_attrs.append(route_encoder[route])
                route_attrs.append(route_encoder[route])

                route_types.append(route_encoder[route].index(1))
                route_types.append(route_encoder[route].index(1))

                if n1 not in cls._instance.routes[route].keys():
                    cls._instance.routes[route][n1] = set()
                cls._instance.routes[route][n1].add(n2)

                if n2 not in cls._instance.routes[route].keys():
                    cls._instance.routes[route][n2] = set()
                cls._instance.routes[route][n2].add(n1)

                cls._instance._db_rows.append({'a': n1, 'b': n2, 'type': route.strip()})

        cls._instance.edge_index = torch.tensor([start_nodes, end_nodes], dtype = torch.long).to(cls._instance.device)
        cls._instance.edge_attr = torch.tensor(route_attrs, dtype = torch.float).to(cls._instance.device)
        cls._instance.edge_type = torch.tensor(route_types).to(cls._instance.device)

        cls._instance.reset()

        return cls._instance


    def reset(self) -> None:
        """ Reset all graph data to default. """

        self.potential_x_pos = list(State.X_STARTER_NODES)

        if not self.use_neo4j:
            return

        self.db = Database()
        if not self.is_db_loaded():
            self.flush_db()
            self.fill_db()


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
        features = [row.copy() for row in self.feature_shape_x]

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

        # 6 : On Mr X's node, how many remaining black tickets does he have? (0 to 5 normalized)
        features[x_node - 1][6] = state.tickets['x']['black'] / 5

        # 7 : On Mr X's node, how many remaining double tickets does he have? (0 to 2 normalized)
        features[x_node - 1][6] = state.tickets['x']['double'] / 2

        # 8 : Do detectives think Mr X might be here? (0 or 1)
        for node in self.get_potential_x_pos(state):
            features[node - 1][8] = 1

        # 9 : Is there a Detective on this node? (0 or 1)
        for node in detectives_pos.values():
            features[node - 1][9] = 1

        # 10: Could a Detective get to this node in the next move? (0 or 1)
        for det in detectives_pos.keys():
            for node in self.get_reachable_neighbors(state, det):
                features[node - 1][10] = 1

        # 11: The number of taxi tickets the Detective on this node has. (0 to 11 normalized)
        # 12: The number of bus tickets the Detective on this node has. (0 to 8 normalized)
        # 13: The number of metro tickets the Detective on this node has. (0 to 4 normalized)
        for det, node in detectives_pos.items():
            features[node - 1][11] = state.tickets[det]['taxi'] / 11
            features[node - 1][12] = state.tickets[det]['bus'] / 8
            features[node - 1][13] = state.tickets[det]['metro'] / 4

        # 14: Round number. (1 to 24 normalized)
        # 15: Number of rounds until Mr. X's next reveal. (0 to 5 normalized)
        rounds_until_reveal = state.get_steps_until_reveal()
        for node in range(self.num_nodes):
            features[node][14] = state.x_step_count / 24
            features[node][15] = rounds_until_reveal / 5

        return features


    def _get_features_for_d(self, state: State, detective: str) -> list[list[int | float]]:
        """ Helper function to get a Detective's GNN input features. """

        other_detectives_pos = {player: pos for player, pos in state.positions.items()
                                if player != detective and player != 'x'}
        current_node = state.positions[detective]
        features = [row.copy() for row in self.feature_shape_d]

        # 0 : Is the Detective on this node? (0 or 1)
        features[current_node - 1][0] = 1

        # 1 : Is there a different Detective on this node? (0 or 1)
        for node in other_detectives_pos.values():
            features[node - 1][1] = 1

        # 2 : Could Mr X be on this node? (0 or 1)
        for node in self.get_potential_x_pos(state):
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
            features[node][10] = state.x_step_count / 24
            features[node][11] = rounds_until_reveal / 5

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
            x = self.get_features_by_player(state, player).to(self.device),
            edge_index = self.edge_index,
            edge_attr = self.edge_attr,
            edge_type = self.edge_type,
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

        if self.use_neo4j:
            return self._get_reachable_by_ticket_neo4j(state, player, ticket, from_node)

        return self._get_reachable_by_ticket_python(state, player, ticket, from_node)


    def _get_reachable_by_ticket_neo4j(self, state: State, player: str, ticket: str,
                                       from_node: int = None) -> list[int]:
        """ Helper function for get_reachable_neighbors_by_ticket. """

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


    def _get_reachable_by_ticket_python(self, state: State, player: str, ticket: str,
                                        from_node: int = None) -> list[int]:
        """ Helper function for get_reachable_neighbors_by_ticket. """

        if state.tickets[player][ticket] == 0:
            return []

        player_station = state.positions[player] if not from_node else from_node

        if ticket == 'black':
            all_neighbors = set()

            for r_dict in self.routes.values():
                if player_station in r_dict.keys():
                    all_neighbors.update(r_dict[player_station])

            return list(all_neighbors)

        return list(self.routes[ticket][player_station]) if player_station in self.routes[ticket] else []


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


    def get_potential_x_pos(self, state: State, update: bool = False) -> list[int]:
        """
        Get Mr. X's potential positions based on his travel logs.

        Arguments:
            state: The current game state.
            update: Whether to update the possible Mr. X positions or return the previously calculated result.

        Returns:
            The nodes that could be occupied by Mr. X.
        """

        if not update:
            return self.potential_x_pos

        last_action = state.travel_logs[-1]
        if isinstance(last_action, int):
            self.potential_x_pos = [last_action]
            return self.potential_x_pos

        new_potential_x_pos = set()
        for node in self.potential_x_pos:
            new_potential_x_pos.update(
                self.get_reachable_neighbors_by_ticket(state, 'x', ticket = last_action, from_node = node)
            )

        self.potential_x_pos = list(new_potential_x_pos)
        return self.potential_x_pos


__all__ = ["Graph"]
