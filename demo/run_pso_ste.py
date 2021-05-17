from parameter_tuning.optimized_nio.mf_optimized_pso_plume import PSO
from applications.plume import GaussianPlumeEAAI
import numpy as np


def main():
    """PSO solving the source term estimation problem."""
    # Model setup
    source = np.array([1500, 8, 10, 5])  # assume source concentration and 3D coordinates
    u, pg_stability = 2, 'F'  # setup environment
    sample_path = r"data/ObservedData.csv"
    # Build model object
    func = GaussianPlumeEAAI(lower=(10, -500, -500, 0), upper=(5000, 500, 500, 10), u=u,
                             pg_stability=pg_stability, sample_path=sample_path)
    # Generate sample observed data
    func.generate_observed_data(source[0], source[1], source[2], source[3])

    # Reverse search source use observed data and PSO (assume unknown the source)
    pso_search_with_recommended_param(func)
    pso_search_with_optimized_param(func)


def pso_search_with_recommended_param(func):
    pso = PSO(func=func, iterations=200, population=30)
    best_sol, best_val = pso.run()
    print("Recommended parameter PSO result: \n", best_sol)


def pso_search_with_optimized_param(func):
    params = [0.32045679, 1.12915509, 3.18318237, 6.36621569]  # MetaWOA(Linear-MF)
    pso = PSO(func=func, iterations=200, population=30,
              w=params[0], c1=params[1], c2=params[2], v_max=params[3])
    best_sol, best_val = pso.run()
    print("MetaWOA(Linear-MF) optimized PSO result: \n", best_sol)


if __name__ == '__main__':
    main()
