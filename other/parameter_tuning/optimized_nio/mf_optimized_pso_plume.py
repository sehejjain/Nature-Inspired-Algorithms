from parameter_tuning.optimized_nio.mf_optimized_nio_base import MFOptimizedNIOBase
from algorithms import ParticleSwarmOptimization
from applications.plume import GaussianPlumeEAAI
from numpy import array
import logging

logging.basicConfig()
logger = logging.getLogger('MFOptimizedPSOPlumeFunc')
logger.setLevel('INFO')


class PSO(ParticleSwarmOptimization):

    def update_velocity(self, velocity, particle_pos, pbest_pos, gbest_pos):
        self.w = 0.8 - (0.8 - 0.1) * self.iter / self.iterations  # override default weight=0.8
        return self.w * velocity + self.c1 * self.Rand.rand(self.population, self.dim) * (pbest_pos - particle_pos) + \
               self.c2 * self.Rand.rand(self.population, self.dim) * (gbest_pos - particle_pos)


class MFOptimizedPSOPlumeFunc(MFOptimizedNIOBase):

    def __init__(self, lower=(0, 0, 0, 0), upper=(1, 10, 10, 20), benchmark=None, fidelity_option='Linear',
                 fixed_fidelity=10, scale=20, population=30):
        dimension = 4
        super(MFOptimizedPSOPlumeFunc, self).__init__(lower, upper, dimension, benchmark, fidelity_option,
                                                      fixed_fidelity, scale, population)
        # Override benchmark to GaussianPlumeEAAI
        if benchmark is None:
            self.source = array([1500, 8, 10, 5])
            self.u, self.pg_stability = 2, 'F'
            self.sample_path = r"data/ObservedData.csv"
            self.benchmark = GaussianPlumeEAAI(lower=(10, -500, -500, 0), upper=(5000, 500, 500, 10), u=self.u,
                                               pg_stability=self.pg_stability, sample_path=self.sample_path)
            self.benchmark.generate_observed_data(self.source[0], self.source[1], self.source[2], self.source[3])

    def get_optimum(self):
        return array([[0.7, 2.0, 2.0, 4.0]]), self.benchmark.get_optimum()[-1]

    def eval(self, params):
        iterations = self.fidelity * self.scale  # default scale=20
        pso = PSO(w=params[0], c1=params[1], c2=params[2], v_max=params[3], func=self.benchmark, iterations=iterations, population=self.population)
        best = pso.run_return_best_val()
        self.eval_count += pso.eval_count
        return best
