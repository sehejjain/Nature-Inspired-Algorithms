from numpy import full, ndarray, array, zeros, inf, power, exp, sum, squeeze
import os


class Benchmark:

    def __init__(self, func_id=1, lower=-100.0, upper=100.0, dimension=2):
        self.func_id = func_id
        if dimension not in [2, 10, 20, 30, 50, 100]:
            raise ValueError("{bench}: Dimension is only defined for [2, 10, 20, 30, 50, 100]".format(bench=self.__class__.__name__))
        self.dimension = dimension
        if isinstance(lower, (float, int)):
            self.lower = full(self.dimension, lower)
            self.upper = full(self.dimension, upper)
        elif isinstance(lower, (ndarray, list, tuple)) and len(lower) == dimension:
            self.lower = array(lower)
            self.upper = array(upper)
        else:
            raise ValueError("{bench}: Type mismatch or Length of bound mismatch with dimension".format(bench=self.__class__.__name__))

        self.base_path = os.path.dirname(__file__)
        self.matrix_data = self.load_matrix_data()
        self.shift_data = self.load_shift_data()
        self.shuffle_data = self.load_shuffle_data()

        self.inf = 1.0e99

    def get_optimum(self):
        return array([zeros(self.dimension)]), 0.0
        pass

    @staticmethod
    def eval(**kwargs):
        return inf
        pass

    # Shift and Rotate
    def shift_rotate(self, x, shift_data, matrix_data, shrink_rate, shift_flag, rotate_flag):
        sr_x = x

        if shift_flag:
            sr_x = self.shift_func(x, shift_data)

        sr_x *= shrink_rate  # shrink to the original search range

        if rotate_flag:
            sr_x = self.rotate_func(sr_x, matrix_data)
        return sr_x

    @staticmethod
    def shift_func(x, shift_data):
        x_shift = x - shift_data
        return x_shift

    @staticmethod
    def rotate_func(x, matrix_data):
        x_rotate = matrix_data.dot(x.T)
        return x_rotate

    def load_matrix_data(self):
        path = os.path.join(self.base_path, r"input_data/M_{func_id}_D{dim}.txt".format(func_id=self.func_id, dim=self.dimension))
        with open(path, 'r') as file:
            lines = file.readlines()
            rows = len(lines)

            data = zeros((rows, self.dimension))
            row = 0
            for line in lines:
                line = line.strip().split()
                data[row, :] = line[:]
                row += 1

        return squeeze(data)

    def load_shift_data(self):
        path = os.path.join(self.base_path, r"input_data/shift_data_{func_id}.txt".format(func_id=self.func_id))

        with open(path, 'r') as file:
            lines = file.readlines()
            rows = len(lines)

            data = zeros((rows, self.dimension))
            row = 0
            for line in lines:
                line = line.strip().split()
                data[row, :] = line[:self.dimension]
                row += 1

        return squeeze(data)

    def load_shuffle_data(self):
        path = os.path.join(self.base_path, r"input_data/shuffle_data_{func_id}_D{dim}.txt".format(func_id=self.func_id, dim=self.dimension))

        with open(path, 'r') as file:
            line = file.readline().strip().split()
        data = array(line, dtype=int)

        return squeeze(data)

    def cf_cal(self, x, fit, shift_data, cf_num, delta, lamb, bias):
        fit = fit * lamb + bias

        w = zeros(cf_num)
        d = len(x)
        for i in range(cf_num):
            shift_x = x - shift_data[i]
            sum1 = power(shift_x, 2).sum()
            if sum1 != 0:
                w[i] = power(1.0 / sum1, 0.5) * exp(-sum1 / 2.0 / d / power(delta[i], 2))
            else:
                w[i] = self.inf
        w_max = w.max()
        w_sum = w.sum()

        if w_max == 0:
            w = full(cf_num, 1)
            w_sum = cf_num

        f = sum(w / w_sum * fit)
        return f


if __name__ == '__main__':
    b = Benchmark(dimension=10)
