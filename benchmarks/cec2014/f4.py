from benchmarks.cec2014.benchmark import Benchmark
from benchmarks.cec2014.basic_functions import rosenbrock_func
import numpy as np


class F4(Benchmark):
    r"""
    B. Multimodal Functions
    F4: Shifted and Rotated Rosenbrock’s Function
    """

    def __init__(self, lower=-100.0, upper=100.0, dimension=2):
        super(F4, self).__init__(4, lower, upper, dimension)

    def get_optimum(self):
        return self.shift_data, 100.0 * self.func_id

    def eval(self, x):
        sr_x = self.shift_rotate(x, self.shift_data, self.matrix_data, 2.048/100.0, True, True)  # shift and rotate
        sr_x += 1.0  # shift to origin
        f = rosenbrock_func(sr_x) + 100 * self.func_id
        return f


if __name__ == '__main__':
    f4 = F4(lower=-100.0, upper=100.0, dimension=10)

    x1 = f4.load_shift_data()
    print(x1)
    print("f4(x1)={}".format(f4.eval(x1)))

    x2 = np.array([0.0] * 10)
    print("f4(x2)={}".format(f4.eval(x2)))

    x3 = np.array([10.0] * 10)
    print("f4(x3)={}".format(f4.eval(x3)))
