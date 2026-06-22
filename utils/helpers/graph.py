class Graph:
    """ Game board representation as a graph. """

    taxi_routes: dict[int, set[int]] = {}
    bus_routes: dict[int, set[int]] = {}
    metro_routes: dict[int, set[int]] = {}
    ferry_routes: dict[int, set[int]] = {}


    def __init__(self, graph_file: str, first_node: int, last_node: int) -> None:
        """
        Build a new graph object from a csv file.

        Arguments:
            graph_file: The csv file where the graph data is stored.
            first_node: The lower-inclusive bound of the nodes.
            last_node: The upper-inclusive bound of the nodes.
        """

        self.routes = {'taxi': self.taxi_routes, 'bus': self.bus_routes,
                       'metro': self.metro_routes, 'ferry': self.ferry_routes}

        for route in self.routes.values():
            for i in range(first_node, last_node + 1):
                route[i] = set()

        with open(graph_file, 'r') as file:
            for line in file.read().split('\n'):
                if not line or line.startswith('#'):
                    continue

                n1, n2, route = line.split(',')
                n1, n2 = int(n1), int(n2)
                route = self.routes[route]

                route[n1].add(n2)
                route[n2].add(n1)


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