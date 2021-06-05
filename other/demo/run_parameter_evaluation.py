from algorithms import ParticleSwarmOptimization as PSO
from benchmarks import Sphere
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    params = [0.10411034, 0.79605184, 2.83067439, 0.20255640]  # MetaGWO(Sigmoid-MF) optimized parameters for PSO solving Sphere-D50
    func = Sphere(dimension=50)  # test function
    rand = np.random.RandomState(seed=1)  # pseudo-random number generator with seed=1
    convergence(params=params, func=func, rand=rand)
    multiple_runs(params=params, func=func, rand=rand)


def convergence(params, func, rand):
    """Convergence test."""
    pso = PSO(func=func, iterations=100, population=30, seed=rand.randint(1, 100),
              w=params[0], c1=params[1], c2=params[2], v_max=params[3])
    pso_cvg = pso.run_return_convergence()  # run and save result
    pso_cvg.plot(title="MetaGWO(Sigmoid-MF) PSO Convergence Curve")  # plot convergence curve
    plt.show()


def multiple_runs(params, func, rand):
    """Multiple runs test."""
    runs = 30  # run times
    pso_30_runs = pd.DataFrame(columns=['MetaGWO(Sigmoid-MF)-PSO'], index=list(range(runs)))  # result
    for i in range(runs):
        pso = PSO(func=func, iterations=100, population=30, seed=rand.randint(1, 100 * runs),
                  w=params[0], c1=params[1], c2=params[2], v_max=params[3])
        pso_30_runs['MetaGWO(Sigmoid-MF)-PSO'].iloc[i] = pso.run_return_best_val()  # run and save result
    pso_30_runs.plot(kind="box", title="MetaGWO(Sigmoid-MF) PSO 30-runs Value")  # plot box
    plt.show()


if __name__ == '__main__':
    main()
