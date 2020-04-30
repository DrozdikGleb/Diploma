import math
import sys
from typing import List

import numpy as np
import networkx as nx
import itertools

from meta_gan.GraphBuilder import GraphBuilder


class Vertex:
    def __init__(self, num: int, values):
        self.num = num
        self.values = values  # Элементы в строчке или столбце(признаки объекта, либо значение признака для каждого объекта)
        self.neighbors = {}


class Edge:
    def __init__(self, num, value, from_vert, to_vert):
        self.num = num
        self.value = value
        self.from_vert = from_vert
        self.to_vert = to_vert


def count_euclidean_distance(a: list, b: list):
    assert len(a) == len(b)
    return math.sqrt(sum((a1 - b1) ** 2 for a1, b1 in zip(a, b)))


def build_complete_graph(verts: List[Vertex]):
    n = len(verts)
    distance_matrix = [[0 for _ in range(n)] for _ in range(n)]
    edges = []
    edge_num = 0
    for i in range(len(verts) - 1):
        cur_vert = verts[i]
        for j in range(i + 1, len(verts)):
            next_vert = verts[j]
            # dist = round(count_euclidean_distance(cur_vert.values, next_vert.values), 2) double
            dist = round(count_euclidean_distance(cur_vert.values, next_vert.values), 2)
            distance_matrix[next_vert.num][cur_vert.num] = dist
            distance_matrix[cur_vert.num][next_vert.num] = dist
            cur_vert.neighbors[next_vert.num] = dist
            next_vert.neighbors[cur_vert.num] = dist
            edges.append(Edge(edge_num, dist, cur_vert.num, next_vert.num))
            edge_num += 1
    return edges, distance_matrix


def swap_vertex(G, i, j):
    mapping = {i : j, j : i}
    H = nx.relabel_nodes(G, mapping)
    return H


def relabel_vert(G, perm):
    mapping = {}
    for i in range(len(perm)):
        mapping[i] = perm[i]
    return nx.relabel_nodes(G, mapping)


def hypercube_brute_force(dim, distance_matrix, verts):
    G = nx.hypercube_graph(dim)
    G = nx.convert_node_labels_to_integers(G)  # get_relabeled_graph(G, len(feature_verts), dim)
    for u, v, d in G.edges(data=True):
        G[u][v]['weight'] = distance_matrix[u][v]
    total_min = sys.float_info.max
    res_G = G
    perms = list(itertools.permutations(G.nodes()))
    res = []
    for i in range(len(perms)):
        total = 0.0
        G = relabel_vert(G, perms[i])
        for u, v, d in G.edges(data=True):
            G[u][v]['weight'] = distance_matrix[u][v]
            total += distance_matrix[u][v]
        res.append(total)
        if total < total_min:
            total_min = total
            res_G = G
    return res_G, total_min, verts


def create_vertices(data):
    obj_verts = []
    feature_verts = []
    for i, obj in enumerate(data):
        obj_verts.append(Vertex(i, obj))
    for i, feature in enumerate(data.T):
        feature_verts.append((Vertex(i, feature)))

    return obj_verts, feature_verts


def main():
    data = np.load('./16_3_9.npy')
    data_0 = data[0]
    graph_builder = GraphBuilder()
    G = graph_builder.build_complete_graph_numpy(data_0)
    return G


if __name__ == '__main__':
    main()

    # ans = count_euclidean_distance(a, b)
