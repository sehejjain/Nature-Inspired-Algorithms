from numpy import pi, cos, exp, array
from benchmarks.benchmark import Benchmark


class Custom(Benchmark):
    """dim: 2"""
    def __init__(self, name, eval, lower=-100, upper=100, dimension=2):
        super(Custom, self).__init__(lower, upper, dimension)
        self.eval = eval
        self.__name__ = name

    def get_optimum(self):
        return array([[pi, pi]]), -1

    
