from benchmarks.cec2014.benchmark import Benchmark
from benchmarks.cec2014.basic_functions import rastrigin_func
import numpy as np


class F9(Benchmark):
    r"""
    B. Multimodal Functions
    F5: Shifted and Rotated Rastrigin’s Function
    """

    def __init__(self, lower=-100.0, upper=100.0, dimension=2):
        super(F9, self).__init__(9, lower, upper, dimension)

    def get_optimum(self):
        return self.shift_data, 100.0 * self.func_id

    def eval(self, x):
        sr_x = self.shift_rotate(x, self.shift_data, self.matrix_data, 5.12/100.0, True, True)  # shift and rotate
        f = rastrigin_func(sr_x) + 100 * self.func_id
        return f


if __name__ == '__main__':
    f9 = F9(lower=-100.0, upper=100.0, dimension=10)

    x1 = f9.load_shift_data()
    print("x1=", x1)
    print("f9(x1)={}".format(f9.eval(x1)))

    x2 = np.array([0.0] * 10)
    print("f9(x2)={}".format(f9.eval(x2)))

    x3 = np.array([10.0] * 10)
    print("f9(x3)={}".format(f9.eval(x3)))
