from algorithms import GreyWolfOptimizer
from parameter_tuning.optimized_nio import MFOptimizedNIOBase
import logging

logging.basicConfig()
logger = logging.getLogger('MFMetaGWO')
logger.setLevel('INFO')


class MFMetaGWO(GreyWolfOptimizer):

    def __init__(self, **kwargs):
        super(MFMetaGWO, self).__init__(**kwargs)
        if not isinstance(self.func, MFOptimizedNIOBase):
            raise ValueError("{optimizer}: optimized_nio should be an instance of MFOptimizedNIOBase".format(
                optimizer=self.__class__.__name__))
    
    def cost_function(self, position):
        self.func.fidelity_update(self.iter, self.iterations)
        self.eval_count += 1
        return super(MFMetaGWO, self).cost_function(position)
