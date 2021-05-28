from algorithms import ParticleSwarmOptimization, WhaleOptimizationAlgorithm, CuckooSearch, GreyWolfOptimizer, FruitFly, SquirrelSearchAlgorithm, KrillHerd, WhaleOptimizationAlgorithm, WaterWaveOptimization
from benchmarks import Ackley, Sphere, Schwefel, Schwefel2, XinSheYangN2, Rosenbrock, Griewank, Levy13, Rastrigin, Bukin6, Easom, Camel3, Shubert, Eggholder, Crossit, Michalewicz, Holdertable
import logging, os
import pandas as pd
import tqdm

for benchmark in [Eggholder, Crossit, Michalewicz, Holdertable]:
    # benchmark = Shubert
    for algo in [ ParticleSwarmOptimization, CuckooSearch, GreyWolfOptimizer, FruitFly, KrillHerd, WhaleOptimizationAlgorithm, WaterWaveOptimization]:

        filename = "Results/"+benchmark.__name__+"/"+algo.__name__+".csv"
        list1 = []


        try:
            orig_df = pd.read_csv(filename)
            orig_df = orig_df.drop(columns=["Unnamed: 0"])
        except:
            pass

        for i in tqdm.trange(10):
            cs = algo(func=benchmark(), iterations=300, debug=False)
            best_sol, best_val = cs.run()
            list1.append(best_val)

        df = pd.DataFrame(list1, columns=["fitness"])
        try:
            df = df.append(orig_df) 
        except :
            pass

        try:
            df.to_csv(filename)
        except:
            os.makedirs("Results/"+benchmark.__name__)
            df.to_csv(filename)

