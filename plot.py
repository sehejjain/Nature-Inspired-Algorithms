from benchmarks import PressureVessel,Stybtang, Schaffer2, Camel6, Ackley, Sphere, Schwefel, Schwefel2, XinSheYangN2, Rosenbrock, Griewank, Levy13, Rastrigin, Bukin6, Easom, Camel3, Shubert, Eggholder, Crossit, Michalewicz, Holdertable

benchmarks = [Easom, Camel3, Eggholder, Sphere, Ackley]
# benchmarks = [Levy13]
for benchmark in benchmarks:
    func = benchmark()
    print(func.get_optimum())
    func.plot(scale=0.32, save_path="plots", show=False)