import networkx as nx
import numpy as np
from typing import List
import math
import subprocess

import torch
from torch_geometric.utils import from_networkx

class Vertex:
    def __init__(self, num: int, values):
        self.num = num
        self.values = values  # Элементы в строчке или столбце(признаки объекта, либо значение признака для каждого объекта)
        self.neighbors = {}

class GraphBuilder:

    def build_complete_graph(self, data):
        data = data.squeeze().detach().numpy()
        return self.build_complete_graph_numpy(data[0]), self.build_complete_graph_numpy(data[1])

    def build_complete_graph_numpy(self, data):
        obj_verts, feature_verts = self.create_vertices(data)
        distance_matrix = self.get_distance_matrix(obj_verts)
        np_distance_matrix = np.array(distance_matrix)
        G = nx.from_numpy_matrix(np_distance_matrix)
        G = from_networkx(G)
        G = self.set_features_to_vertices(G, obj_verts)
        return G


    def build_hypercube(self, data):
        graph_file = "input.txt"
        max_time = 1
        cmd = "java -jar Hypercube.jar search %s  ./results/ %d" % (graph_file, max_time)
        subprocess.run(cmd, check=True, shell=True)
        pass

    def set_features_to_vertices(self, G, vertices):
        x = []
        for cur_vert in vertices:
            x.append(cur_vert.values)
        G.x = torch.tensor(x)
        return G

    def create_vertices(self, data):
        obj_verts = []
        feature_verts = []
        for i, obj in enumerate(data):
            obj_verts.append(Vertex(i, obj))
        for i, feature in enumerate(data.T):
            feature_verts.append((Vertex(i, feature)))

        return obj_verts, feature_verts

    def count_euclidean_distance(self, a: list, b: list):
        assert len(a) == len(b)
        return math.sqrt(sum((a1 - b1) ** 2 for a1, b1 in zip(a, b)))

    def get_distance_matrix(self, verts: List[Vertex]):
        n = len(verts)
        distance_matrix = [[0 for _ in range(n)] for _ in range(n)]
        edges = []
        edge_num = 0
        for i in range(len(verts) - 1):
            cur_vert = verts[i]
            for j in range(i + 1, len(verts)):
                next_vert = verts[j]
                dist = self.count_euclidean_distance(cur_vert.values, next_vert.values)
                distance_matrix[next_vert.num][cur_vert.num] = dist
                distance_matrix[cur_vert.num][next_vert.num] = dist
                cur_vert.neighbors[next_vert.num] = dist
                next_vert.neighbors[cur_vert.num] = dist
                # edges.append(Edge(edge_num, dist, cur_vert.num, next_vert.num))
                # edge_num += 1
        return distance_matrix
