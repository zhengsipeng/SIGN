import numpy as np


class Graph():
    def __init__(self,
                 num_node=17+2,
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.num_node = num_node
        self.max_hop = max_hop
        self.dilation = max_hop
        self.get_edge()
        self.hop_dis = self.get_hop_distance(self.num_node,
                                        self.edge,
                                        max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout='coco'):
        # edge is a list of [child, parent] paris
        # among them, the former 17 nodes are human-pose keynodes, and the latter 2 nodes are object nodes
        if layout == 'coco':
            # pose-pose connection
            neighbor_1base = [(16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13), (6, 7), (8, 6),
                              (9, 7), (10, 8), (11, 9), (2, 3), (2, 1), (3, 1), (4, 2), (5, 3), (4, 6), (5, 7)]
            # object-pose connection, including nose, both hands and both feet
            neighbor_2base = [(18, 16), (18, 17), (18, 10), (18, 11), (18, 1),
                              (19, 16), (19, 17), (19, 17), (19, 17), (19, 17)]
            self_link = [(i, i) for i in range(self.num_node)]
            if self.num_node == 19:
                neighbor_base = neighbor_1base + neighbor_2base
            elif self.num_node == 17:
                neighbor_base = neighbor_1base
            else:
                raise NotImplementedError
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_base]

            self.edge = self_link + neighbor_link
            self.center = 0
        else:
            raise ValueError('Do not exist this layout')

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        self.ori_A = adjacency
        normalize_adjacency = self.normalize_digraph(adjacency)
        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError('Do not exist this strategy')

    def get_hop_distance(self, num_node, edge, max_hop=1):
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[i, j] = 1
            A[j, i] = 1

        # compute hop steps
        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)  # digr matrix
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = np.dot(A, Dn)
        return AD

    def normalize_undigraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-0.5)
        DAD = np.dot(np.dot(Dn, A), Dn)
        return DAD
