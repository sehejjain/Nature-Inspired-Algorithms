from algorithms import ParticleSwarmOptimization as PSO
from benchmarks import Sphere
import numpy as np
import matplotlib.pyplot as plt


def main():
    func = Sphere(dimension=50)  # cost function
    print("Sphere: \n", np.squeeze(func.get_optimum()[0]), func.get_optimum()[1])

    # use recommended parameter(default)
    pso_rec = PSO(func=func, iterations=100, population=30)
    best_sol, best_val = pso_rec.run()
    print("Recommended parameter PSO result: \n", best_sol, best_val)
    pso_rec.iter_solution['Fitness'].plot(title="Shpere-D50: use recommended parameter PSO")
    plt.show()

    # use Clerc's parameter
    params = [0.7298, 1.49618, 1.49618, func.upper.max()]
    pso_clerc = PSO(func=func, iterations=100, population=30,
                    w=params[0], c1=params[1], c2=params[2], v_max=params[3])
    best_sol, best_val = pso_clerc.run()
    print("Clerc's parameter PSO result: \n", best_sol, best_val)
    pso_clerc.iter_solution['Fitness'].plot(title="Shpere-D50: use Clerc's parameter PSO")
    plt.show()

    # use multi-fidelity tuned parameter
    params = [0.10411034, 0.79605184, 2.83067439, 0.20255640]  # Meta-GWO(FCF:Sigmoid)
    pso_mf = PSO(func=func, iterations=100, population=30,
                 w=params[0], c1=params[1], c2=params[2], v_max=params[3])
    best_sol, best_val = pso_mf.run()
    print("Multi-fidelity tuned parameter PSO result: \n", best_sol, best_val)
    pso_mf.iter_solution['Fitness'].plot(title="Shpere-D50: use multi-fidelity tuned parameter PSO")
    plt.show()


if __name__ == '__main__':
    main()
