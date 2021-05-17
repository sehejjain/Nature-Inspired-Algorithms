from benchmarks.cec2014.benchmark import Benchmark
from benchmarks.cec2014.basic_functions import bent_cigar_func
import numpy as np


class F2(Benchmark):
    r"""
    A. Unimodal Functions
    F2: Rotated Bent Cigar Function
    """

    def __init__(self, lower=-100.0, upper=100.0, dimension=2):
        super(F2, self).__init__(2, lower, upper, dimension)

    def get_optimum(self):
        return self.shift_data, 100.0 * self.func_id

    def eval(self, x):
        sr_x = self.shift_rotate(x, self.shift_data, self.matrix_data, 1.0, True, True)  # shift and rotate
        f = bent_cigar_func(sr_x) + 100 * self.func_id
        return f


if __name__ == '__main__':
    f2 = F2(lower=-100.0, upper=100.0, dimension=10)

    x1 = f2.load_shift_data()
    print(x1)
    print("f2(x1)={}".format(f2.eval(x1)))

    x2 = np.array([0.0] * 10)
    print("f2(x2)={}".format(f2.eval(x2)))

    x3 = np.array([10.0] * 10)
    print("f2(x3)={}".format(f2.eval(x3)))
