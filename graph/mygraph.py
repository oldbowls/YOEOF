import sys

# sys.path.insert(0, '')
# sys.path.extend(['../'])

from graph import tools

num_node = 11
self_link = [(i, i) for i in range(num_node)]
inward = [(0, 10), (10, 1), (10, 2), (10, 1), (1, 3), (3, 5), (2, 4), (4, 6), (10, 9), (7, 9), (9, 8)]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class AdjMatrixGraph:
    def __init__(self):
        self.edge = neighbor
        self.num_nodes = num_node
        self.self_loop = [(i, i) for i in range(self.num_nodes)]  # 自连接矩阵
        self.A_binary = tools.get_adjacency_matrix(self.edge, self.num_nodes)  # A
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edge + self.self_loop, self.num_nodes)
        self.A = tools.normalize_adjacency_matrix(self.A_binary)
