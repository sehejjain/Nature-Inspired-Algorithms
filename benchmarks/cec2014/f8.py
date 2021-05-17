from benchmarks.cec2014.benchmark import Benchmark
from benchmarks.cec2014.basic_functions import rastrigin_func
import numpy as np


class F8(Benchmark):
    r"""
    B. Multimodal Functions
    F5: Shifted Rastrigin’s Function
    """

    def __init__(self, lower=-100.0, upper=100.0, dimension=2):
        super(F8, self).__init__(8, lower, upper, dimension)

    def get_optimum(self):
        return self.shift_data, 100.0 * self.func_id

    def eval(self, x):
        sr_x = self.shift_rotate(x, self.shift_data, self.matrix_data, 5.12/100.0, True, False)  # shift and rotate
        f = rastrigin_func(sr_x.T) + 100 * self.func_id
        return f


if __name__ == '__main__':
    f8 = F8(lower=-100.0, upper=100.0, dimension=10)

    x1 = f8.load_shift_data()
    print("x1=", x1)
    print("f8(x1)={}".format(f8.eval(x1)))

    x2 = np.array([0.0] * 10)
    print("f8(x2)={}".format(f8.eval(x2)))

    x3 = np.array([10.0] * 10)
    print("f8(x3)={}".format(f8.eval(x3)))
