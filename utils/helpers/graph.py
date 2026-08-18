from dotenv import load_dotenv
from neo4j import GraphDatabase, RoutingControl
import os



print(f'loaded database: {os.getenv("blabla")}')


class Graph:
    """ Game board representation as a graph. """

    def __init__(self, graph_file: str) -> None:
        """
        Build the graph object from a csv file using Neo4j.

        Database Setup:
            Initializing an object of this class attempts to connect to a Neo4j Aura instance.
            Make sure you have the appropriate credentials inside the `.env` file:
            `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`.

        Arguments:
            graph_file: The csv file where the graph data is stored.
        """

        load_dotenv()
        self.db_uri = os.getenv('NEO4J_URI')
        self.db_auth = (os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
        self.db_name = os.getenv('NEO4J_DATABASE')

        if self.db_uri is None or self.db_auth is None or self.db_name is None:
            raise Exception('One or more Neo4j Aura credentials are missing from environment variables.')

        self.db_driver = GraphDatabase.driver(self.db_uri, auth = self.db_auth)

        with self.db_driver as driver:
            driver.verify_connectivity()

        # TODO: Check if graph is loaded into database, load it if not
        # TODO: Replace existing logic with Neo4j queries

        # self.routes = {'taxi': self.taxi_routes, 'bus': self.bus_routes,
        #                'metro': self.metro_routes, 'ferry': self.ferry_routes}
        #
        # for route in self.routes.values():
        #     for i in range(first_node, last_node + 1):
        #         route[i] = set()
        #
        # with open(graph_file, 'r') as file:
        #     for line in file.read().split('\n'):
        #         if not line or line.startswith('#'):
        #             continue
        #
        #         n1, n2, route = line.split(',')
        #         n1, n2 = int(n1), int(n2)
        #         route = self.routes[route]
        #
        #         route[n1].add(n2)
        #         route[n2].add(n1)


    def get_neighbors_by_route(self, node: int, route: str) -> set[int]:
        """
        Get a specific node's neighbors linked via the given route.

        Arguments:
             node: The node.
             route: The transport route ("taxi", "bus", "metro", or "ferry").

        Returns:
            A set of nodes connected to the specified node through the given route type.
        """

        return self.routes[route.lower()][node]


    def get_neighbors(self, node: int) -> set[int]:
        """
        Get a specific node's neighbors.

        Arguments:
             node: The node.

        Returns:
            A set of all neighboring nodes to the specified node regardless of route type.
        """

        neighbors = self.taxi_routes[node]
        neighbors.update(self.bus_routes[node])
        neighbors.update(self.metro_routes[node])
        neighbors.update(self.ferry_routes[node])

        return neighbors


__all__ = ["Graph"]
