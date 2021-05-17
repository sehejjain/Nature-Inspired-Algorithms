from benchmarks.cec2014.benchmark import Benchmark
from benchmarks.cec2014.basic_functions import elliptic_func,bent_cigar_func,discus_func, rosenbrock_func
import numpy as np


class F23(Benchmark):
    r"""
    D. Composition Functions
    F23: Composition Function 1
    """

    def __init__(self, lower=-100.0, upper=100.0, dimension=2):
        super(F23, self).__init__(23, lower, upper, dimension)

    def get_optimum(self):
        return self.shift_data, 100.0 * self.func_id

    def eval(self, x):
        r_flag = True
        n = 5
        delta = np.array([10, 20, 30, 40, 50])
        lamb = np.array([1, 1e-6, 1e-26, 1e-6, 1e-6])
        bias = np.array([0, 100, 200, 300, 400])
        d = self.dimension

        shift_data = []
        matrix_data = []
        for i in range(n):
            shift_data.append(self.shift_data[i])
            matrix_data.append(self.matrix_data[i*d:(i+1)*d])

        fit = np.zeros(n)

        sr_x = self.shift_rotate(x, shift_data[0], matrix_data[0], 2.048/100.0, True, r_flag)
        sr_x += 1.0
        fit[0] = rosenbrock_func(sr_x)

        sr_x = self.shift_rotate(x, shift_data[1], matrix_data[1], 1.0, True, r_flag)
        fit[1] = elliptic_func(sr_x)

        sr_x = self.shift_rotate(x, shift_data[2], matrix_data[2], 1.0, True, r_flag)
        fit[2] = bent_cigar_func(sr_x)

        sr_x = self.shift_rotate(x, shift_data[3], matrix_data[3], 1.0, True, r_flag)
        fit[3] = discus_func(sr_x)

        sr_x = self.shift_rotate(x, shift_data[4], matrix_data[4], 1.0, True, False)
        fit[4] = elliptic_func(sr_x)

        f = self.cf_cal(x, fit, shift_data, n, delta, lamb, bias) + 100 * self.func_id
        return f


if __name__ == '__main__':
    f23 = F23(lower=-100.0, upper=100.0, dimension=10)
    print(f23.load_shuffle_data().size)
    print(f23.load_shift_data().size)

    x1 = f23.load_shift_data()[0]
    print(x1)
    print("f23(x1)={}".format(f23.eval(x1)))

    x2 = np.array([0.0] * 10)
    print("f23(x2)={}".format(f23.eval(x2)))

    x3 = np.array([10.0] * 10)
    print("f23(x3)={}".format(f23.eval(x3)))
