import numpy as np

from modifiedLMGAN.GraphBuilder import GraphBuilder


def main():
    h = 64
    dp = [[0 for x in range(h)] for y in range(h)]
    k = 1
    for i in range(h):
        for j in range(i + 1, h):
            dp[i][j] = dp[j][i] = k
            k += 1
    [print(*line) for line in dp]


def main2():
    graph_builder = GraphBuilder()
    data = np.load("loader/datasets/dprocessed_16_64_2/6_0_2.npy")
    graph_builder.build_hypercube(data[0])


def main3():
    graph_builder = GraphBuilder()
    graph_builder.get_best_hypercube("./modifiedLMGAN/results")

if __name__ == '__main__':
    main3()

