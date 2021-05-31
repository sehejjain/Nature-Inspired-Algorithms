from numpy import zeros, pi, cos, exp, sqrt, array
from benchmarks.benchmark import Benchmark
import numpy as np


class PressureVessel(Benchmark):
    """dim: n"""

    def __init__(self, lower=[1.125, 0.625, 10**-8, 10**-8], upper=[12.5, 12.5, 240, 240], dimension=4):
        super(PressureVessel, self).__init__(lower, upper, dimension)

    def get_optimum(self):
        return array([1.125, 0.625, 58.29016, 43.69266]), 7197.729

    @staticmethod
    def eval(sol):
        x1 = sol[0]
        x2 = sol[1]
        x3 = sol[2]
        x4 = sol[3]
        if 0.0193*x3 - x3 > 0 or 0.00954*x3 - x3 > 0:
            return np.Inf
        if np.pi*(x3**2)*x4*-1 - (4/3)*np.pi*(x3**3) + 1296000 > 0:
            return np.Inf
        if x4 - 240 > 0:
            return np.inf

        t1 = 0.6224*x1*x3*x4
        t2 = 1.7781*x2*(x3**2)
        t3 = 3.1611*(x1**2)*x4
        t4 = 19.84*(x1**2)*x3
        return t1+t2+t3+t4


if __name__ == '__main__':
    ackley = PressureVessel()
    print(ackley.get_optimum())
    ackley.plot(scale=0.32)
