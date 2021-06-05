from algorithms import ParticleSwarmOptimization, WhaleOptimizationAlgorithm, CuckooSearch, GreyWolfOptimizer, FruitFly, SquirrelSearchAlgorithm, KrillHerd, WhaleOptimizationAlgorithm, WaterWaveOptimization, ArtificialBeeColony
from benchmarks import PressureVessel, Stybtang, Schaffer2, Camel6, Ackley, Sphere, Schwefel, Schwefel2, XinSheYangN2, Rosenbrock, Griewank, Levy13, Rastrigin, Bukin6, Easom, Camel3, Shubert, Eggholder, Crossit, Michalewicz, Holdertable, FreqModulated
import logging, os
import pandas as pd
import tqdm
algos = [ArtificialBeeColony]
for benchmark in [Sphere]:
    # benchmark = Shubert
    for algo in algos:

        filename = "Results/"+benchmark.__name__+"/"+algo.__name__+".csv"
        # progname = "Progress/"+benchmark.__name__+"/"+algo.__name__+".csv"
        
        list1 = []


        try:
            orig_df = pd.read_csv(filename)
            orig_df = orig_df.drop(columns=["Unnamed: 0"])

        except:
            pass
        progress = []
        for i in tqdm.trange(10):
            cs = algo(func=benchmark(), iterations=500, debug=False)
            best_sol, best_val = cs.run()
            # progress.append(cs.progress)
            list1.append(best_val)

        df = pd.DataFrame(list1, columns=["fitness"])
        # progDF = pd.DataFrame(progress).transpose()
        try:
            df = df.append(orig_df) 
        except :
            pass

        try:
            df.to_csv(filename)
            
        except:
            os.makedirs("Results/"+benchmark.__name__)
            df.to_csv(filename)
        # try:
        #     progDF.to_csv(progname)
            
        # except:
        #     os.makedirs("Progress/"+benchmark.__name__)
        #     progDF.to_csv(progname)

