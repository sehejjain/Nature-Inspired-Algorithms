from benchmarks import Benchmark
from numpy import array, arange, dtype, pi, exp, mgrid, fabs
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator
import logging

logging.basicConfig()
logger = logging.getLogger("GaussianPlumeEAAI")
logger.setLevel('INFO')


class GaussianPlumeEAAI(Benchmark):

    def __init__(self, lower=(10, -500, -500, 0), upper=(5000, 500, 500, 30), dimension=4, u=2, pg_stability='F', sample_path=None):
        """
        Observation Point: (Q, X, Y, Z)

        Center: (Q0, X0, Y0, Z0)
        Q0 in [10, 5000]; X0, Y0 in [-512, 512]; Z0 in [0, 30]

        u: Wind speed
        pgStability: A-F

        """
        super(GaussianPlumeEAAI, self).__init__(lower, upper, dimension)
        self.u = u
        self.pg_stability = pg_stability

        self.observed_data_path = sample_path
        # self.observed_data_path = os.path.join(os.path.dirname(__file__), r"data/ObservedData.csv")
        if self.observed_data_path:
            try:
                self.observed_data = pd.read_csv(self.observed_data_path, header=0, index_col=0)
            except FileNotFoundError as e:
                logger.error("ObservedData not exist. Error: " + str(e))
                raise e
        else:
            logger.warning("Not specify an observed data file.")

    def get_optimum(self):
        """Center: (Q0, X0, Y0, Z0)"""
        return array([[1500, 8, 10, 5]]), 0

    def eval(self, sol):
        """Cost Function: sum(f_i - f^hat)"""

        concn_measured = self.observed_data['concentration']
        concn_model = self.calculate_concentration(self.observed_data['X'], self.observed_data['Y'], self.observed_data['Z'], sol[0], sol[1], sol[2], sol[3], self.u, self.pg_stability)
        error = (concn_measured - concn_model) ** 2
        return error.sum()

    @staticmethod
    def sigma_y(pg_stability, x):
        if pg_stability == 'A':
            return 0.22 * x * (1 + 0.0001 * x) ** (-0.5)
        elif pg_stability == 'B':
            return 0.16 * x * (1 + 0.0001 * x) ** (-0.5)
        elif pg_stability == 'C':
            return 0.11 * x * (1 + 0.0001 * x) ** (-0.5)
        elif pg_stability == 'D':
            return 0.08 * x * (1 + 0.0001 * x) ** (-0.5)
        elif pg_stability == 'E':
            return 0.06 * x * (1 + 0.0001 * x) ** (-0.5)
        else:
            return 0.04 * x * (1 + 0.0001 * x) ** (-0.5)

    @staticmethod
    def sigma_z(pg_stability, x):
        if pg_stability == 'A':
            return 0.2 * x
        elif pg_stability == 'B':
            return 0.12 * x
        elif pg_stability == 'C':
            return 0.08 * x * (1 + 0.0002 * x) ** (-0.5)
        elif pg_stability == 'D':
            return 0.06 * x * (1 + 0.0015 * x) ** (-0.5)
        elif pg_stability == 'E':
            return 0.03 * x * (1 + 0.0003 * x) ** (-1)
        else:
            return 0.016 * x * (1 + 0.0003 * x) ** (-1)

    def calculate_concentration(self, x, y, z, q0, x0, y0, z0, u, pg_stability):
        part1 = (q0 / (2 * pi * self.sigma_y(pg_stability, (x - x0)) * self.sigma_z(pg_stability, (x - x0)) * u)) * exp(
            -0.5 * ((y - y0) ** 2) / (self.sigma_y(pg_stability, (x - x0)) ** 2))
        part2 = exp(-((z - z0) ** 2) / (2 * (self.sigma_z(pg_stability, (x - x0)) ** 2))) + exp(
            -((z + z0) ** 2) / (2 * (self.sigma_z(pg_stability, (x - x0)) ** 2)))

        return part1 * part2

    def generate_observed_data(self, q0, x0, y0, z0):
        z = 3
        obs_points = array([[250, 10, z], [250, 20, z], [250, 30, z],
                            [350, 10, z], [350, 20, z], [350, 30, z],
                            [450, 10, z], [450, 20, z], [450, 30, z],
                            [250, -10, z], [250, -20, z], [250, -30, z],
                            [350, -10, z], [350, -20, z], [350, -30, z],
                            [450, -10, z], [450, -20, z], [450, -30, z]], dtype=dtype(float))
        obs_data = pd.DataFrame(index=arange(obs_points.shape[0]), columns=['X', 'Y', 'Z', 'concentration'], dtype=dtype(float))

        concn = self.calculate_concentration(obs_points[:, 0], obs_points[:, 1], obs_points[:, 2], q0, x0, y0, z0, self.u, self.pg_stability)
        obs_data[['X', 'Y', 'Z']], obs_data['concentration'] = obs_points, concn
        self.observed_data = obs_data
        self.observed_data_path = self.observed_data_path or r"data/ObservedData.csv"
        obs_data.to_csv(self.observed_data_path)

    def temp_visualization(self, q0, x0, y0, z0, u, pg_stability):
        u = u if u else self.u
        pg_stability = pg_stability if pg_stability else self.pg_stability

        X, Y = mgrid[20:500:2, 2:30:0.5]
        Z0 = self.calculate_concentration(X, Y, 0, q0, x0, y0, z0, u, pg_stability)
        Z1 = self.calculate_concentration(X, Y, 3, q0, x0, y0, z0, u, pg_stability)
        Z2 = self.calculate_concentration(X, Y, 8, q0, x0, y0, z0, u, pg_stability)
        Z3 = self.calculate_concentration(X, Y, 10, q0, x0, y0, z0, u, pg_stability)

        fig1 = plt.figure()
        ax1a = fig1.add_subplot(221, projection='3d')
        ax1a.plot_surface(X, Y, Z0, cmap='winter', cstride=2, rstride=2)
        ax1a.set_xlabel("x(m)")
        ax1a.set_ylabel("y(m)")
        ax1a.set_zlabel("z(g/m3)")
        ax1a.set_zlim(0, 5)
        for spine in ax1a.spines.values(): spine.set_visible(False)

        ax1b = fig1.add_subplot(222, projection='3d')
        ax1b.plot_surface(X, Y, Z1, cmap='winter', cstride=2, rstride=2)
        ax1b.set_xlabel("x(m)")
        ax1b.set_ylabel("y(m)")
        ax1b.set_zlabel("z(g/m3)")
        ax1b.set_zlim(0, 8)
        for spine in ax1b.spines.values(): spine.set_visible(False)

        ax1c = fig1.add_subplot(223, projection='3d')
        ax1c.plot_surface(X, Y, Z2, cmap='winter', cstride=2, rstride=2)
        ax1c.set_xlabel("x(m)")
        ax1c.set_ylabel("y(m)")
        ax1c.set_zlabel("z(g/m3)")
        ax1c.set_zlim(0, 5)
        for spine in ax1c.spines.values(): spine.set_visible(False)

        ax1d = fig1.add_subplot(224, projection='3d')
        ax1d.plot_surface(X, Y, Z3, cmap='winter', cstride=2, rstride=2)
        ax1d.set_xlabel("x(m)")
        ax1d.set_ylabel("y(m)")
        ax1d.set_zlabel("z(g/m3)")
        ax1d.set_zlim(0, 5)
        for spine in ax1d.spines.values(): spine.set_visible(False)
        plt.savefig('output/test003.png')
        plt.close()

        extent = (-3, 4, -4, 3)
        # Set levels to avoid truncation error
        levels = arange(0, 10, 0.3)
        norm = cm.colors.Normalize(vmax=abs(Z0).max(), vmin=-abs(Z0).max())
        cmap = cm.PRGn
        fig2, _ax2 = plt.subplots(nrows=2, ncols=2)
        fig2.subplots_adjust(hspace=0.3)
        ax2 = _ax2.flatten()
        cset01 = ax2[0].contourf(X, Y, Z0, levels, norm=norm, cmap=cm.get_cmap(cmap, len(levels) - 1))
        cset02 = ax2[0].contour(X, Y, Z0, cset01.levels, colors='g')
        # for c in cset2.collections: c.set_linestyle('solid')
        # ax2[0].set_title('subfig 1')
        fig2.colorbar(cset01, ax=ax2[0])
        cset11 = ax2[1].contourf(X, Y, Z1, levels, norm=norm, cmap=cm.get_cmap(cmap, len(levels) - 1))
        cset12 = ax2[1].contour(X, Y, Z1, cset11.levels, colors='g')
        # for c in cset2.collections: c.set_linestyle('solid')
        # ax2[0].set_title('subfig 1')
        fig2.colorbar(cset11, ax=ax2[1])
        cset21 = ax2[2].contourf(X, Y, Z2, levels, norm=norm, cmap=cm.get_cmap(cmap, len(levels) - 1))
        cset22 = ax2[2].contour(X, Y, Z2, cset21.levels, colors='g')
        # for c in cset2.collections: c.set_linestyle('solid')
        # ax2[0].set_title('subfig 1')
        fig2.colorbar(cset21, ax=ax2[2])
        cset31 = ax2[3].contourf(X, Y, Z3, levels, norm=norm, cmap=cm.get_cmap(cmap, len(levels) - 1))
        cset32 = ax2[3].contour(X, Y, Z3, cset31.levels, colors='g')
        # for c in cset2.collections: c.set_linestyle('solid')
        # ax2[0].set_title('subfig 1')
        fig2.colorbar(cset31, ax=ax2[3])
        for spine in ax2[0].spines.values(): spine.set_visible(False)

        plt.savefig('output/test004.png')
        plt.close()


if __name__ == '__main__':
    U, pgStability = 2, 'F'
    sample_path = r"data/ObservedData.csv"
    gp = GaussianPlumeEAAI(u=U, pg_stability=pgStability, sample_path=sample_path)

    # generate observed data
    Q0, X0, Y0, Z0,  = 1500, 8, 10, 5
    gp.generate_observed_data(Q0, X0, Y0, Z0)
    gp.temp_visualization(Q0, X0, Y0, Z0, U, pgStability)

    # calculate cost function
    center_point_sample = array([1500, 10, 10, 5])
    cf = gp.eval(center_point_sample)
    print(cf)
