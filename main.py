from algorithms import ParticleSwarmOptimization, WhaleOptimizationAlgorithm, CuckooSearch, GreyWolfOptimizer, FruitFly, SquirrelSearchAlgorithm, KrillHerd, WhaleOptimizationAlgorithm, WaterWaveOptimization, ArtificialBeeColony
from benchmarks import PressureVessel, Stybtang, Schaffer2, Camel6, Ackley, Sphere, Schwefel, Schwefel2, XinSheYangN2, Rosenbrock, Griewank, Levy13, Rastrigin, Bukin6, Easom, Camel3, Shubert, Eggholder, Crossit, Michalewicz, Holdertable, FreqModulated
import logging, os
import pandas as pd
import tqdm

benchmark = Sphere

logging.basicConfig()
logger = logging.getLogger('cs')
logger.setLevel('INFO')
opti = ArtificialBeeColony(func=benchmark(), iterations=500, debug=True, numb_bees = 5)
best_sol, best_val = opti.run()
logger.info("best sol:{sol}, best val:{val}".format(sol=best_sol, val=best_val))
