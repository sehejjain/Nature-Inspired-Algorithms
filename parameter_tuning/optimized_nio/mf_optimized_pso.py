from parameter_tuning.optimized_nio.mf_optimized_nio_base import MFOptimizedNIOBase
from algorithms import ParticleSwarmOptimization
from numpy import array
import logging

logging.basicConfig()
logger = logging.getLogger('MFOptimizedPSOFunc')
logger.setLevel('INFO')


class MFOptimizedPSOFunc(MFOptimizedNIOBase):

    def __init__(self, lower=(0, 0, 0, 0), upper=(1, 10, 10, 20), benchmark=None, fidelity_option='Linear', fixed_fidelity=10, scale=10, population=30):
        dimension = 4
        super(MFOptimizedPSOFunc, self).__init__(lower, upper, dimension, benchmark, fidelity_option, fixed_fidelity, scale, population)

    def get_optimum(self):
        return array([[0.7, 2.0, 2.0, 4.0]]), self.benchmark.get_optimum()[-1]

    def eval(self, params):
        iterations = self.fidelity * self.scale
        pso = ParticleSwarmOptimization(w=params[0], c1=params[1], c2=params[2], v_max=params[3], func=self.benchmark, iterations=iterations, population=self.population)
        best = pso.run_return_best_val()
        self.eval_count += pso.eval_count
        return best
