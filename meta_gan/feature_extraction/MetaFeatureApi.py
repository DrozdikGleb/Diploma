from abc import ABC, abstractmethod
import numpy as np


class MetaFeature(ABC):

    def __init__(self, features_size: int, instances_size: int, threshold: int = 10000):
        self.features = features_size
        self.instances = instances_size
        self.threshold = threshold

    @abstractmethod
    def getLength(self) -> int:
        pass

    @abstractmethod
    def getMeta(self, zero_in: np.ndarray, one_in: np.ndarray) -> np.ndarray:
        pass

    def mean(self, data_in) -> np.ndarray:
        non_nan = np.nan_to_num(data_in)
        non_nan[non_nan > self.threshold] = 0.0
        non_nan[non_nan < -self.threshold] = 0.0
        return np.mean(non_nan)

    @staticmethod
    def data(zero_in: np.ndarray, one_in: np.ndarray) -> np.ndarray:
        return np.append(zero_in, one_in, axis=0)

    def labels(self) -> np.ndarray:
        return np.append(np.zeros(self.instances), np.ones(self.instances))
