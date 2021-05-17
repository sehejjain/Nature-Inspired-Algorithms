from benchmarks.cec2014.benchmark import Benchmark
from benchmarks.cec2014.basic_functions import discus_func
import numpy as np


class F3(Benchmark):
    r"""
    A. Unimodal Functions
    F3: Rotated Discus Function
    """

    def __init__(self, lower=-100.0, upper=100.0, dimension=2):
        super(F3, self).__init__(3, lower, upper, dimension)

    def get_optimum(self):
        return self.shift_data, 100.0 * self.func_id

    def eval(self, x):
        sr_x = self.shift_rotate(x, self.shift_data, self.matrix_data, 1.0, True, True)  # shift and rotate
        f = discus_func(sr_x) + 100 * self.func_id
        return f


if __name__ == '__main__':
    f3 = F3(lower=-100.0, upper=100.0, dimension=10)

    x1 = f3.load_shift_data()
    print(x1)
    print("f3(x1)={}".format(f3.eval(x1)))

    x2 = np.array([0.0] * 10)
    print("f3(x2)={}".format(f3.eval(x2)))

    x3 = np.array([10.0] * 10)
    print("f3(x3)={}".format(f3.eval(x3)))
