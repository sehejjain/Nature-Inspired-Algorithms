import abc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from numpy import inf, arange, meshgrid, vectorize, full, zeros, array, ndarray


class Benchmark(metaclass=abc.ABCMeta):

    def __init__(self, lower, upper, dimension):
        self.dimension = dimension
        if isinstance(lower, (float, int)):
            self.lower = full(self.dimension, lower)
            self.upper = full(self.dimension, upper)
        elif isinstance(lower, (ndarray, list, tuple)) and len(lower) == dimension:
            self.lower = array(lower)
            self.upper = array(upper)
        else:
            raise ValueError("{bench}: Type mismatch or Length of bound mismatch with dimension".format(bench=self.__class__.__name__))

    def get_optimum(self):
        return array([zeros(self.dimension)]), 0.0
        pass

    @staticmethod
    def eval(**kwargs):
        return inf
        pass

    @staticmethod
    def __2d_func(x, y, f): return f((x, y))

    def plot(self, scale=None, save_path=None):
        if not scale:
            scale = abs(self.upper[0] / 100)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        # Make data.
        X, Y = arange(self.lower[0], self.upper[0], scale), arange(self.lower[1], self.upper[1], scale)
        X, Y = meshgrid(X, Y)
        func = self.eval
        Z = vectorize(self.__2d_func)(X, Y, func)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.6, cmap=plt.get_cmap('coolwarm'), linewidth=0, antialiased=False)

        # Customize the axis.
        ax.set_xlim(self.lower[0], self.upper[0])
        ax.set_ylim(self.lower[1], self.upper[1])
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        # fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()
        if save_path:
            file = save_path + r'/{benchmark}.png'.format(benchmark=self.__class__.__name__)
            fig.savefig(file, dpi=200)
        plt.clf()
        plt.close()
