import numpy as np
import scipy.stats as stats

from feature_extraction.MetaFeatureApi import MetaFeature


class StatisticalMeta(MetaFeature):
    def getLength(self) -> int:
        kurtosis = 1
        pearson = 1
        skew = 1
        result = sum([kurtosis, pearson, skew])
        return result

    def getMeta(self, zero_in: np.ndarray, one_in: np.ndarray) -> np.ndarray:
        data_in = self.data(zero_in, one_in)
        kurtosis = self.mean(stats.kurtosis(data_in, 0))

        pearsons = []
        for i in range(self.features):
            for j in range(i + 1, self.features):
                pearsons.append(stats.pearsonr(data_in[:, i], data_in[:, j])[0])
        pearson = self.mean(pearsons)

        skew = self.mean(stats.skew(data_in, axis=0))

        return np.array([kurtosis, pearson, skew])


if __name__ == '__main__':
    meta = StatisticalMeta(4, 3)
    arr = np.array([[1, 1, 1, 1],
                    [0.2, 0.3, 0, 0],
                    [0.2, 0, 0, 1]])
    print(meta.getMeta(arr, arr))
    print(meta.getLength())
