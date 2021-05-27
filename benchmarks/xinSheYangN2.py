from numpy import array, zeros
import numpy as np
from benchmarks.benchmark import Benchmark


class XinSheYangN2(Benchmark):
    """dim: n"""
    def __init__(self, lower=-5, upper=5, dimension=2):
        super(XinSheYangN2, self).__init__(lower, upper, dimension)

    def get_optimum(self):
        return array([zeros(self.dimension)]), 0.0

    @staticmethod
    def eval(X):
        X = np.array(X)
        d = X.shape[0]
        res = np.sum(np.abs(X))*np.exp(-np.sum(np.sin(X**2)))
        return res
