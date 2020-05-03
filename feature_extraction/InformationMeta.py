import warnings

import numpy as np
import scipy.stats as stats
import sklearn.metrics as metrics

from feature_extraction.MetaFeatureApi import MetaFeature

warnings.filterwarnings("ignore", category=RuntimeWarning)


class InformationMeta(MetaFeature):
    def getLength(self) -> int:
        maxMutInfo = 1
        meanMutInfo = 1
        entropy = 1
        noiseSignal = 1
        classEnt = 1
        result = sum([maxMutInfo, meanMutInfo, entropy, noiseSignal, classEnt])
        return result

    def getMeta(self, zero_in: np.ndarray, one_in: np.ndarray) -> np.ndarray:
        data_in = self.data(zero_in, one_in)
        label_in = self.labels()
        max_mutual_info = 0.0
        for i in range(self.features):
            max_mutual_info = max(max_mutual_info,
                                  metrics.normalized_mutual_info_score(data_in[:, i], label_in,
                                                                       average_method='geometric'))
        max_mutual_info = self.mean(np.array([max_mutual_info]))

        mean_mutual_info = []
        for i in range(self.features):
            mean_mutual_info.append(
                metrics.normalized_mutual_info_score(data_in[:, i], label_in, average_method='geometric'))
        mean_mutual_info = self.mean(mean_mutual_info)

        entropies = []
        for i in range(self.features):
            entropies.append(stats.entropy(data_in[:, i]))
        entropy = self.mean(entropies)

        s_n_ration = self.mean(self.signaltonoise(data_in))

        entopies_cl = [stats.entropy(label_in)]
        cl_enptropy = self.mean(np.array(entopies_cl))

        return np.array([max_mutual_info, mean_mutual_info, entropy, s_n_ration, cl_enptropy])

    @staticmethod
    def signaltonoise(a, axis=0, ddof=0):
        a = np.asanyarray(a)
        m = a.mean(axis)
        sd = a.std(axis=axis, ddof=ddof)
        return np.where(sd == 0, 0, m / sd)


if __name__ == '__main__':
    meta = InformationMeta(4, 3)
    arr = np.array([[1, 1, 0.2, 1],
                    [0.5, 0.3, 0.3, 0],
                    [0.2, 0, 0, 1]])
    print(meta.getMeta(arr, arr))
    print(meta.getLength())
