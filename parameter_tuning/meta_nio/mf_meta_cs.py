from algorithms import CuckooSearch
from parameter_tuning.optimized_nio import MFOptimizedNIOBase
import logging

logging.basicConfig()
logger = logging.getLogger('MFMetaCS')
logger.setLevel('INFO')


class MFMetaCS(CuckooSearch):

    def __init__(self, **kwargs):
        super(MFMetaCS, self).__init__(**kwargs)
        if not isinstance(self.func, MFOptimizedNIOBase):
            raise ValueError("{optimizer}: optimized_nio should be an instance of MFOptimizedNIOBase".format(
                optimizer=self.__class__.__name__))

    def cost_function(self, position):
        self.func.fidelity_update(self.iter, self.iterations)
        self.eval_count += 1
        return super(MFMetaCS, self).cost_function(position)


if __name__ == '__main__':
    from benchmarks import Sphere
    from parameter_tuning.optimized_nio import MFOptimizedCSFunc

    benchmark = Sphere(dimension=10)
    optimized_cs = MFOptimizedCSFunc(benchmark=benchmark, fidelity_option='ReLU', scale=10)
    optimizer_cs = MFMetaCS(func=optimized_cs, iterations=200)
    best_sol, best_val = optimizer_cs.run()
    logger.info("best sol:{sol}, best val:{val}".format(sol=best_sol, val=best_val))
