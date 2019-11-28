import numpy as np


class Graph():
    def __init__(self, num_node=17+2, strategy='spatial', max_hop=1):
        self.num_node = num_node
        self.max_hop = max_hop
        self.dilation = max_hop
        self.get_edge()
        self.hop_dis = self.get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout='coco'):
        # edge is a list of [child, parent] paris
        # among them, the former 17 nodes are human-pose keynodes, and the latter 2 nodes are object nodes
        if layout == 'coco':
            # skeleton-skeleton connection
            neighbor_1base = self.get_neighbor(1)

            # skeleton-object connection, including nose, both hands and both feet
            neighbor_2base = self.get_neighbor(2)

            # bodypart-skeleton and bodypart-bodypart connection,
            # bodypart id from 18 to 23, correspongding to head, body, l-r hand, l-r foot
            neighbor_3base = self.get_neighbor(3)

            # object-skeleton and object-bodypart connection, object id is 24
            neighbor_4base = self.get_neighbor(4)

            # bodypart-bodypart and object-bodypart with non skeleton
            neighbor_5base = self.get_neighbor(5)
            neighbor_6base = self.get_neighbor(6)

            self_link = [(i, i) for i in range(self.num_node)]

            if self.num_node == 17:
                neighbor_base = neighbor_1base
            elif self.num_node == 17+2:
                neighbor_base = neighbor_1base + neighbor_2base
            elif self.num_node == 17+6:
                neighbor_base = neighbor_1base + neighbor_3base
            elif self.num_node == 17+6+1:
                neighbor_base = neighbor_1base + neighbor_3base + neighbor_4base
            elif self.num_node == 6:
                neighbor_base = neighbor_5base
            elif self.num_node == 7:
                neighbor_base = neighbor_5base + neighbor_6base
            else:
                raise NotImplementedError
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_base]

            self.edge = self_link + neighbor_link
            self.center = 0
        else:
            raise ValueError('Do not exist this layout')

    def get_neighbor(self, baseid):
        if baseid == 1:  # skeletons-skeletons
            return [(16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13), (6, 7), (8, 6),
                    (9, 7), (10, 8), (11, 9), (2, 3), (2, 1), (3, 1), (4, 2), (5, 3), (1, 6), (1, 7)]
        elif baseid == 2:  # skeletons-objects
            return [(18, 1), (18, 10), (18, 11), (18, 16), (18, 17),
                    (19, 1), (19, 10), (19, 11), (19, 16), (19, 17),
                    (18, 19)]
        elif baseid == 3:  # bodypart-bodypart & bodypart-skeletons
            return [(18, 1), (18, 2), (18, 3), (18, 4), (18, 5),
                    (19, 6), (19, 7), (19, 12), (19, 13),
                    (20, 6), (20, 8), (20, 10), (21, 7), (21, 9), (21, 11),
                    (22, 12), (22, 14), (22, 16), (23, 13), (23, 15), (23, 17),
                    (18, 19), (19, 20), (19, 21), (19, 22), (19, 23)]
        elif baseid == 4:  # object-bodypart
            return [(24, 1), (24, 10), (24, 11), (24, 16), (24, 17),
                    (24, 18), (24, 19), (24, 20), (24, 21), (24, 22), (24, 23)]
        elif baseid == 5:  # non-ske, bodypart-bodypart
            return [(1, 2), (2, 3), (2, 4), (2, 5), (2, 6)]
        elif baseid == 6:  # non-ske, object-bodypart
            return [(7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6)]
        else:
            raise NotImplementedError

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
