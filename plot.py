# from benchmarks import PressureVessel,Stybtang, Schaffer2, Camel6, Ackley, Sphere, Schwefel, Schwefel2, XinSheYangN2, Rosenbrock, Griewank, Levy13, Rastrigin, Bukin6, Easom, Camel3, Shubert, Eggholder, Crossit, Michalewicz, Holdertable

# benchmarks = [Easom, Camel3, Eggholder, Sphere, Ackley]
# # benchmarks = [Levy13]
# for benchmark in benchmarks:
#     func = benchmark()
#     print(func.get_optimum())
#     func.plot(scale=0.32, save_path="plots", show=False)

from cec2017.hybrid import f14
from cec2017.utils import surface_plot

for func in [f14]:
    try:
        surface_plot(func, dimension=10)
    except:
        pass