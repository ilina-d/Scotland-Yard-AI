from utils.helpers import Graph


g = Graph('graphs/test.csv', 1, 6)
for i in range(1, 6 + 1):

    for j in g.get_neighbors_by_route(i, 'taxi'):
        print(f'Node {i} connected to {j} via taxi.')

    for j in g.get_neighbors_by_route(i, 'bus'):
        print(f'Node {i} connected to {j} via bus.')

    for j in g.get_neighbors_by_route(i, 'metro'):
        print(f'Node {i} connected to {j} via metro.')

    for j in g.get_neighbors_by_route(i, 'ferry'):
        print(f'Node {i} connected to {j} via ferry.')

    print('---------------------------------------')
    input()
