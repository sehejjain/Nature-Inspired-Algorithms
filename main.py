from algorithms import ParticleSwarmOptimization, WhaleOptimizationAlgorithm, CuckooSearch, GreyWolfOptimizer, FruitFly, SquirrelSearchAlgorithm, KrillHerd, WhaleOptimizationAlgorithm, WaterWaveOptimization, ArtificialBeeColony
from benchmarks import Custom, PressureVessel, Stybtang, Schaffer2, Camel6, Ackley, Sphere, Schwefel, Schwefel2, XinSheYangN2, Rosenbrock, Griewank, Levy13, Rastrigin, Bukin6, Easom, Camel3, Shubert, Eggholder, Crossit, Michalewicz, Holdertable, FreqModulated
import logging, os
import pandas as pd
import tqdm
from cec2017.composition import f21, f22, f23, f24, f25, f26, f27, f28, f29, f30
from cec2017.hybrid import f11, f12, f13, f14, f15, f16, f17, f18, f19, f20
from cec2017.simple import f1, f2, f3, f4, f5, f6, f7, f8, f9, f10

# benchmarks = [Custom(name= custom_benchmark.__name__,eval=custom_benchmark, lower=-100, upper=100, dimension=2) for custom_benchmark in [f3]]
# benchmarks = [Custom(name = f1.__name__, eval = f1, lower=-100, upper=100, dimension=2)]
benchmarks = [Schwefel, Camel6, Schaffer2, Levy13]
algos = [KrillHerd, WhaleOptimizationAlgorithm, WaterWaveOptimization, ArtificialBeeColony]
for benchmark in benchmarks:
    try:
        for algo in algos:
            filename = "Results/"+benchmark.__name__+"/"+algo.__name__+".csv"
            progname = "Progress/"+benchmark.__name__+"/"+algo.__name__+".csv"
                
            list1 = []
            
            try:
                orig_df = pd.read_csv(filename)
                orig_df = orig_df.drop(columns=["Unnamed: 0"])
            except:
                pass
            progress = []
            for i in tqdm.trange(10):
                try:
                    cs = algo(func=benchmark, iterations=350, debug=False, numb_bees=20)
                except:
                    cs = algo(func=benchmark(), iterations=350, debug=False, numb_bees=20)
                best_sol, best_val = cs.run()
                progress.append(cs.progress)
                list1.append(best_val)

            df = pd.DataFrame(list1, columns=["fitness"])
            progDF = pd.DataFrame(progress).transpose()
            try:
                df = df.append(orig_df) 
            except :
                pass

            # try:
            #     df.to_csv(filename)
                
            # except:
            #     os.makedirs("Results/"+benchmark.__name__)
            #     df.to_csv(filename)
            try:
                progDF.to_csv(progname)
                
            except:
                os.makedirs("Progress/"+benchmark.__name__)
                progDF.to_csv(progname)

            # except:
            #     print(benchmark.__name__)
    except:
        pass