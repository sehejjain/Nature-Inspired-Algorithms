from algorithms import ParticleSwarmOptimization, WhaleOptimizationAlgorithm, CuckooSearch, GreyWolfOptimizer, FruitFly, SquirrelSearchAlgorithm, KrillHerd
from benchmarks import Ackley, Sphere
import logging
import pandas as pd

logging.basicConfig()
logger = logging.getLogger('cs')
logger.setLevel('INFO')
# cs = GreyWolfOptimizer(func=Ackley(), iterations=500, debug=True)
# best_sol, best_val = cs.run()
# logger.info("best sol:{sol}, best val:{val}".format(sol=best_sol, val=best_val))

list1 = []
import tqdm
filename = "sphere_cs.csv"
try:
    orig_df = pd.read_csv(filename)
    orig_df = orig_df.drop(columns=["Unnamed: 0"])
except:
    pass

for i in range(1):
    cs = SquirrelSearchAlgorithm(func=Ackley(), iterations=500, debug=False)
    best_sol, best_val = cs.run()
    # list1.append(best_val)
    # best_sol, best_val = cs.run()
    logger.info("best sol:{sol}, best val:{val}".format(sol=best_sol, val=best_val))

# import pandas as pd

# df = pd.DataFrame(list1, columns=["fitness"])
# try:
#     df = df.append(orig_df) 
# except :
#     pass

# df.to_csv(filename)

