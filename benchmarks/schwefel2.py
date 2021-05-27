from numpy import array, full, sqrt, sin, abs
from benchmarks.benchmark import Benchmark


class Schwefel2(Benchmark):
    """dim: n"""
    def __init__(self, lower=-500, upper=500, dimension=2):
        super(Schwefel2, self).__init__(lower, upper, dimension)

    def get_optimum(self):
        return array([full(self.dimension, 420.9687)]), 2.545567497236334e-05

    @staticmethod
    def eval(chromosome):
        part1 = 0.0
        part2 = 1.0
        for c in chromosome:
            part1 += abs(c)
            part2 *= abs(c)
        return part1+part2
