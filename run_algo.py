from algorithms import ParticleSwarmOptimization, WhaleOptimizationAlgorithm, CuckooSearch, GreyWolfOptimizer, FruitFly, SquirrelSearchAlgorithm, KrillHerd, WhaleOptimizationAlgorithm, WaterWaveOptimization, ArtificialBeeColony
from benchmarks import Benchmark, PressureVessel, Stybtang, Schaffer2, Camel6, Ackley, Sphere, Schwefel, Schwefel2, XinSheYangN2, Rosenbrock, Griewank, Levy13, Rastrigin, Bukin6, Easom, Camel3, Shubert, Eggholder, Crossit, Michalewicz, Holdertable, FreqModulated
import logging, os
import pandas as pd
import numpy as np
from cec2017.functions import f1
# from cec2017.basic import bent_cigar


func = f1
class CustomBenchmark(Benchmark):
    '''
    Set the domain of the function
    '''
    def __init__(self, lower = -100, upper=100, dimension = 2):
        super(CustomBenchmark, self).__init__(lower, upper, dimension)
    
    def get_optimum(self):
        '''
        Set best known optimum here
        '''
        return np.array([1.125, 0.625, 58.29016, 43.69266]), 7197.729
    @staticmethod
    def eval(sol):
        '''
        define custom method here
        '''
        return func(sol)

logging.basicConfig()
logger = logging.getLogger('cs')
logger.setLevel('INFO')
opti = ParticleSwarmOptimization(func=CustomBenchmark(), iterations=500, debug=False)
best_sol, best_val = opti.run()
logger.info("best sol:{sol}, best val:{val}".format(sol=best_sol, val=best_val))

