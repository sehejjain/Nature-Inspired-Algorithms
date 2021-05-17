from benchmarks.cec2014 import Benchmark
from benchmarks.cec2014 import elliptic_func, rastrigin_func, schwefel_func
import numpy as np


class F17(Benchmark):
    r"""
    C. Hybrid Functions
    F17: Hybrid Function 1
    """

    def __init__(self, lower=-100.0, upper=100.0, dimension=2):
        super(F17, self).__init__(17, lower, upper, dimension)

    def get_optimum(self):
        return self.shift_data, 100.0 * self.func_id

    def eval(self, x):
        sr_x = self.shift_rotate(x,  self.shift_data, self.matrix_data, 1.0, True, True)  # shift and rotate
        y = sr_x[self.shuffle_data - 1]

        n = 3
        p = [0.3, 0.3, 0.4]
        d = len(y)

        x1n = int(np.ceil(p[0] * d))
        x2n = int(np.ceil(p[1] * d))

        y1 = y[0:x1n]
        y2 = y[x1n:x1n + x2n]
        y3 = y[x1n + x2n:d]

        shift_data1 = self.shift_data[0:x1n]
        shift_data2 = self.shift_data[x1n:x1n + x2n]
        shift_data3 = self.shift_data[x1n + x2n:d]

        matrix_data1 = self.matrix_data[0][0:x1n]
        matrix_data2 = self.matrix_data[1][x1n:x1n + x2n]
        matrix_data3 = self.matrix_data[2][x1n + x2n:d]

        # Modified Schwefel's Function f9
        sr_y1 = self.shift_rotate(y1,  shift_data1, matrix_data1, 1000.0/100.0, False, False)
        g1 = schwefel_func(sr_y1)
        # Rastrigin’s Function f8
        sr_y2 = self.shift_rotate(y2,  shift_data2, matrix_data2, 5.12/100.0, False, False)
        g2 = rastrigin_func(sr_y2)
        # High Conditioned Elliptic Function f1
        sr_y3 = self.shift_rotate(y3,  shift_data3, matrix_data3, 1.0, False, False)
        g3 = elliptic_func(sr_y3)

        f = g1 + g2 + g3 + 100 * self.func_id
        return f


if __name__ == '__main__':
    f17 = F17(lower=-100.0, upper=100.0, dimension=10)

    x1 = f17.load_shift_data()
    print(x1)
    print("f17(x1)={}".format(f17.eval(x1)))

    x2 = np.array([0.0] * 10)
    print("f17(x2)={}".format(f17.eval(x2)))

    x3 = np.array([10.0] * 10)
    print("f17(x3)={}".format(f17.eval(x3)))
