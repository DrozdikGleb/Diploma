import networkx as nx
import numpy as np
from typing import List
import os
import math
import re
import subprocess
from collections import defaultdict
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
from threading import Timer

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
        return self.build_hypercube(data[0]), self.build_hypercube(data[1])

    def build_complete_graph_numpy(self, data):
        obj_verts, feature_verts = self.create_vertices(data)
        distance_matrix = self.get_distance_matrix(obj_verts)
        np_distance_matrix = np.array(distance_matrix)
        return self.make_pytorch_graph(np_distance_matrix, obj_verts)




    def build_hypercube(self, data):
        obj_verts, _ = self.create_vertices(data)
        dist_matrix = self.get_distance_matrix(obj_verts)
        np_distance_matrix = self.build_hypercube_from_dist_matrix(dist_matrix)
        return self.make_pytorch_graph(np_distance_matrix, obj_verts)

    def make_pytorch_graph(self, np_distance_matrix, verts):
        G = nx.from_numpy_matrix(np_distance_matrix)
        G = from_networkx(G)
        G = self.set_features_to_vertices(G, verts)
        return G

    def build_hypercube_from_dist_matrix(self, dist_matrix):
        graph_file_path = "./input.txt"
        hypercube_path = "./results"
        self.save_to_file(dist_matrix, graph_file_path)
        max_time = 1
        cmd = "java -jar Hypercube.jar search %s %s %d" % (graph_file_path, hypercube_path,
                                                           max_time)
        self.run_process_with_timeout(cmd, 30)
        min_res, hypercube_verts = self.get_best_hypercube(hypercube_path)
        self.build_hypercube_matrix(dist_matrix, hypercube_verts)

    def build_hypercube_matrix(self, dist_matrix, verts):
        vert_num = len(verts)
        hypercube_matrix = [[0 for _ in range(vert_num)] for _ in range(vert_num)]
        rank = int(math.log2(vert_num))
        hypercube_neighbors = defaultdict(list)
        for i in range(vert_num):
            for j in range(rank):
                if (i ^ (1 << j)) > i:
                    hypercube_neighbors[i + 1].append((i ^ (1 << j)) + 1)
                    hypercube_neighbors[(i ^ (1 << j)) + 1].append(i + 1)
        for cur_vert in range(1, len(verts) + 1):
            cur_list = hypercube_neighbors[cur_vert]
            cur_vert = verts[cur_vert - 1] - 1
            for nei in cur_list:
                cur_nei = verts[nei - 1] - 1
                hypercube_matrix[cur_vert][cur_nei] = hypercube_matrix[cur_nei][cur_vert] = \
                    dist_matrix[cur_nei][cur_vert] / 100.0

        return np.array(hypercube_matrix)
        #self.draw_hypercube(np.array(hypercube_matrix))

    def draw_hypercube(self, np_distance_matrix):
        G = nx.from_numpy_matrix(np_distance_matrix)
        pos = nx.spring_layout(G)
        plt.figure(figsize=(20, 16))
        nx.draw(G, pos)
        # nx.draw_networkx_labels(G, pos, labels, font_size=16)
        nx.draw_networkx_edge_labels(G, pos)
        plt.draw()



    def get_best_hypercube(self, path):
        min_len = 0
        with open(os.path.join(path, "result.txt")) as file:
            line = file.readline()
            line = re.split(r"\s(?![^\[]*\])", line)
            min_len = int(line[0])
            res = line[1][1:-1]
            hypercube_verts = list(map(int, res.split(', ')))
        return min_len, hypercube_verts

    def run_process_with_timeout(self, cmd, timeout_sec):
        try:
            subprocess.call(cmd, timeout=timeout_sec, shell=True)
        except subprocess.TimeoutExpired:
            print("Not finished")
        # proc = Popen(shlex.split(cmd), stdout=PIPE, stderr=PIPE)
        # timer = Timer(timeout_sec, proc.kill)
        # try:
        #     timer.start()
        #     stdout, stderr = proc.communicate()
        # finally:
        #     timer.cancel()

    def save_to_file(self, dist_matrix, file_location):
        vert_numbers = len(dist_matrix)
        rank = int(math.log2(vert_numbers))
        with open(file_location, "w+") as input_file:
            input_file.write("%d %d\n" % (rank, vert_numbers))
            for i in range(len(dist_matrix)):
                cur_line = " ".join(map(str, dist_matrix[i]))
                cur_line += "\n"
                input_file.write(cur_line)



    @staticmethod
    def set_features_to_vertices(G, vertices):
        x = []
        for cur_vert in vertices:
            x.append(cur_vert.values)
        G.x = torch.tensor(x)
        return G

    @staticmethod
    def create_vertices(data):
        obj_verts = []
        feature_verts = []
        for i, obj in enumerate(data):
            obj_verts.append(Vertex(i, obj))
        for i, feature in enumerate(data.T):
            feature_verts.append((Vertex(i, feature)))

        return obj_verts, feature_verts

    @staticmethod
    def count_euclidean_distance(a: list, b: list):
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
                #dist = int(round(self.count_euclidean_distance(cur_vert.values, next_vert.values), 2) * 100)
                distance_matrix[next_vert.num][cur_vert.num] = dist
                distance_matrix[cur_vert.num][next_vert.num] = dist
                cur_vert.neighbors[next_vert.num] = dist
                next_vert.neighbors[cur_vert.num] = dist
                # edges.append(Edge(edge_num, dist, cur_vert.num, next_vert.num))
                # edge_num += 1
        return distance_matrix
