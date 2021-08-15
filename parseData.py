from algorithms import ParticleSwarmOptimization, WhaleOptimizationAlgorithm, CuckooSearch, GreyWolfOptimizer, FruitFly, SquirrelSearchAlgorithm, KrillHerd, WhaleOptimizationAlgorithm, WaterWaveOptimization, ArtificialBeeColony
from benchmarks import PressureVessel,Stybtang, Schaffer2, Camel6, Ackley, Sphere, Schwefel, Schwefel2, XinSheYangN2, Rosenbrock, Griewank, Levy13, Rastrigin, Bukin6, Easom, Camel3, Shubert, Eggholder, Crossit, Michalewicz, Holdertable, FreqModulated, Custom
import logging, os
import pandas as pd
import tqdm
from cec2017.composition import f21, f22, f23, f24, f25, f26, f27, f28, f29, f30
from cec2017.hybrid import f11, f12, f13, f14, f15, f16, f17, f18, f19, f20
from cec2017.simple import f1, f2, f3, f4, f5, f6, f7, f8, f9, f10


benchmarks = [Custom(name= custom_benchmark.__name__,eval=custom_benchmark, lower=-100, upper=100, dimension=2) for custom_benchmark in [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30]]
algos = [ParticleSwarmOptimization, WhaleOptimizationAlgorithm, CuckooSearch, GreyWolfOptimizer, FruitFly, SquirrelSearchAlgorithm, KrillHerd, WhaleOptimizationAlgorithm, WaterWaveOptimization, ArtificialBeeColony]

list1 = []
# benchmarks = [Rosenbrock, Griewank, Levy13, Rastrigin, Bukin6, Easom, Camel3, Shubert, Eggholder, Crossit, Michalewicz, Holdertable]
# benchmarks = [FreqModulated]
for benchmark in benchmarks:
    main_df = pd.DataFrame()
    for algo in algos:
        filename = "Results/new/"+benchmark.__name__+"/"+algo.__name__+".csv"
        df = pd.read_csv(filename)
        main_df[algo.__name__] = df["fitness"]
    main_df.to_csv("Results/compiled/"+benchmark.__name__+".csv")