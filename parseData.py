from algorithms import ParticleSwarmOptimization, WhaleOptimizationAlgorithm, CuckooSearch, GreyWolfOptimizer, FruitFly, SquirrelSearchAlgorithm, KrillHerd, WhaleOptimizationAlgorithm, WaterWaveOptimization
from benchmarks import PressureVessel,Stybtang, Schaffer2, Camel6, Ackley, Sphere, Schwefel, Schwefel2, XinSheYangN2, Rosenbrock, Griewank, Levy13, Rastrigin, Bukin6, Easom, Camel3, Shubert, Eggholder, Crossit, Michalewicz, Holdertable, FreqModulated
import logging, os
import pandas as pd
import tqdm

algos = [ParticleSwarmOptimization, CuckooSearch, GreyWolfOptimizer, FruitFly, KrillHerd, WhaleOptimizationAlgorithm, WaterWaveOptimization]

list1 = []
# benchmarks = [Rosenbrock, Griewank, Levy13, Rastrigin, Bukin6, Easom, Camel3, Shubert, Eggholder, Crossit, Michalewicz, Holdertable]
benchmarks = [FreqModulated]
for benchmark in benchmarks:
    main_df = pd.DataFrame()
    for algo in algos:
# algo = ParticleSwarmOptimization
        filename = "Results/"+benchmark.__name__+"/"+algo.__name__+".csv"
        df = pd.read_csv(filename)
        main_df[algo.__name__] = df["fitness"]
    main_df.to_csv("Results/compiled/"+benchmark.__name__+".csv")