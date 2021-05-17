from benchmarks.cec2014.benchmark import Benchmark
from benchmarks.cec2014.basic_functions import schwefel_func
import numpy as np


class F10(Benchmark):
    r"""
    B. Multimodal Functions
    F5: Shifted Schwefel’s Function
    """

    def __init__(self, lower=-100.0, upper=100.0, dimension=2):
        super(F10, self).__init__(10, lower, upper, dimension)

    def get_optimum(self):
        return self.shift_data, 100.0 * self.func_id

    def eval(self, x):
        sr_x = self.shift_rotate(x, self.shift_data, self.matrix_data, 1000.0/100.0, True, False)  # shift and rotate
        f = schwefel_func(sr_x.T) + 100 * self.func_id
        return f


if __name__ == '__main__':
    f10 = F10(lower=-100.0, upper=100.0, dimension=10)

    x1 = f10.load_shift_data()
    print("x1=", x1)
    print("f10(x1)={}".format(f10.eval(x1)))

    x2 = np.array([0.0] * 10)
    print("f10(x2)={}".format(f10.eval(x2)))

    x3 = np.array([10.0] * 10)
    print("f10(x3)={}".format(f10.eval(x3)))
