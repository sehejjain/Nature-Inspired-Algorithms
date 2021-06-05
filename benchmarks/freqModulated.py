from numpy import zeros, pi, cos, exp, sqrt, array, sin
import sys
sys.path.insert(1, '../')
from benchmarks.benchmark import Benchmark


class FreqModulated(Benchmark):
    """dim: n"""

    def __init__(self, lower=-6.4, upper=6.35, dimension=6):
        super(FreqModulated, self).__init__(lower, upper, dimension)

    def get_optimum(self):
        return array([zeros(self.dimension)]), 0.0

    @staticmethod
    def eval(sol):
        a1 = sol[0]
        w1 = sol[1]
        a2 = sol[2]
        w2 = sol[3]
        a3 = sol[4]
        w3 = sol[5]
        sum = 0
        θ = 2*pi/100
        for t in range(100):
            y_t = a1*sin(w1*t*θ + a2*sin(w2*t*θ + a3*sin(w3*t*θ)))
            y0_t = (1.0)*sin((5.0)*t*θ - (1.5)*sin((4.8)*t*θ + (2.0)*sin((4.9)*t*θ)))
            sum+=(y_t - y0_t)**2
        return sum


if __name__ == '__main__':
    freq = FreqModulated()
    print(freq.get_optimum())
    freq.plot(scale=0.32)
