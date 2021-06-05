from functools import wraps
from time import time
from benchmarks import Sphere
from parameter_tuning.meta_nio import MFMetaGWO
from parameter_tuning.optimized_nio import MFOptimizedPSOFunc


def main():
    func = Sphere(dimension=50)  # cost function
    parameter_tuning(func=func)
    multi_fidelity_parameter_tuning(func=func)


def timer(func):
    """Timer for log time cost."""
    @wraps(func)
    def wrapper_timer(*args, **kwargs):
        print(f'[Function: {func.__name__!r} start...]')
        tic = time()
        try:
            return func(*args, **kwargs)
        finally:
            toc = time()
            run_time = toc - tic
            print(f'[Function: {func.__name__!r} finished in {run_time:.4f} secs]\n')
    return wrapper_timer


@timer
def parameter_tuning(func):
    """Non-MF"""
    optimized_pso = MFOptimizedPSOFunc(benchmark=func, fidelity_option='Fixed', scale=10, population=30)  # Fixed method means fidelity never change
    meta_gwo = MFMetaGWO(func=optimized_pso, iterations=100, population=30)
    solution = meta_gwo.run()
    params = solution[0]  # get optimized parameters of PSO
    eval_counts = optimized_pso.eval_count  # total evaluation times of cost function
    print("MetaGWO(Non-MF) - PSO: \n", "Parameters:", params)
    print(" Evaluation Nums:", eval_counts)


@timer
def multi_fidelity_parameter_tuning(func):
    """Sigmoid-MF"""
    fidelity_ctrl_func = 'Sigmoid'
    optimized_pso = MFOptimizedPSOFunc(benchmark=func, fidelity_option=fidelity_ctrl_func, scale=10, population=30)
    meta_gwo = MFMetaGWO(func=optimized_pso, iterations=100, population=30)
    solution = meta_gwo.run()
    params = solution[0]
    eval_counts = optimized_pso.eval_count
    print("MetaGWO(Sigmoid-MF) - PSO: \n", "Parameters:", params)
    print(" Evaluation Nums:", eval_counts)


if __name__ == '__main__':
    main()
