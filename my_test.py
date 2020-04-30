from collections import defaultdict
from functools import partial


def main():
    h = 16
    dp = [[0 for x in range(h)] for y in range(h)]
    k = 1
    for i in range(h):
        for j in range(i + 1, h):
            dp[i][j] = dp[j][i] = k
            k += 1
    [print(*line) for line in dp]


if __name__ == '__main__':
    main()

