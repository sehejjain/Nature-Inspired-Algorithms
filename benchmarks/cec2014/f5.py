from benchmarks.cec2014.benchmark import Benchmark
from benchmarks.cec2014.basic_functions import ackley_func
import numpy as np


class F5(Benchmark):
    r"""
    B. Multimodal Functions
    F5: Shifted and Rotated Ackley’s Function
    """

    def __init__(self, lower=-100.0, upper=100.0, dimension=2):
        super(F5, self).__init__(5, lower, upper, dimension)

    def get_optimum(self):
        return self.shift_data, 100.0 * self.func_id

    def eval(self, x):
        sr_x = self.shift_rotate(x, self.shift_data, self.matrix_data, 1.0, True, True)  # shift and rotate
        f = ackley_func(sr_x) + 100 * self.func_id
        return f


if __name__ == '__main__':
    f5 = F5(lower=-100.0, upper=100.0, dimension=10)

    x1 = f5.load_shift_data()
    print(x1)
    print("f5(x1)={}".format(f5.eval(x1)))

    x2 = np.array([0.0] * 10)
    print("f5(x2)={}".format(f5.eval(x2)))

    x3 = np.array([10.0] * 10)
    print("f5(x3)={}".format(f5.eval(x3)))
