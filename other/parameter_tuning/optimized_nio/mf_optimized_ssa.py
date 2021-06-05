from parameter_tuning.optimized_nio.mf_optimized_nio_base import MFOptimizedNIOBase
from algorithms import SquirrelSearchAlgorithm
from numpy import array
import logging

logging.basicConfig()
logger = logging.getLogger('MFOptimizedSSAFunc')
logger.setLevel('INFO')


class MFOptimizedSSAFunc(MFOptimizedNIOBase):

    def __init__(self, lower=(1, 0.1, 16), upper=(10, 0.9, 37), benchmark=None, fidelity_option='Linear', fixed_fidelity=10, scale=10, population=30):
        dimension = 3
        super(MFOptimizedSSAFunc, self).__init__(lower, upper, dimension, benchmark, fidelity_option, fixed_fidelity, scale, population)

    def get_optimum(self):
        return array([[1.9, 0.1, 18]]), self.benchmark.get_optimum()[-1]

    def eval(self, params):
        iterations = self.fidelity * self.scale
        ssa = SquirrelSearchAlgorithm(Gc=params[0], Pdp=params[1], sf=params[2], func=self.benchmark, iterations=iterations, population=self.population)
        best = ssa.run_return_best_val()
        self.eval_count += ssa.eval_count
        return best
