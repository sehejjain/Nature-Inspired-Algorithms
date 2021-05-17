import numpy as np


def elliptic_func(x):
    """1. High Conditioned Elliptic Function"""
    f = 0.0
    d = len(x)
    for i in range(d):
        f += np.power(10.0, 6.0 * i / (d - 1)) * x[i] ** 2
    return f


def bent_cigar_func(x):
    """2. Bent Cigar Function"""
    sum = 0
    for i in range(1, len(x)):
        sum += x[i] ** 2
    f = x[0] ** 2 + np.power(10, 6) * sum
    return f


def discus_func(x):
    """3. Discus Function"""
    sum = 0
    for i in range(1, len(x)):
        sum += x[i] ** 2
    f = np.power(10, 6) * x[0] ** 2 + sum
    return f


def rosenbrock_func(x):
    """4. Rosenbrock’s Function"""
    f = 0
    for i in range(len(x)-1):
        f += 100 * (x[i] ** 2 - x[i+1]) ** 2 + (x[i] - 1) ** 2
    return f


def ackley_func(x):
    """5. Ackley’s Function"""
    c = 2.0 * np.pi
    d = len(x)

    sum1 = 0.0
    sum2 = 0.0
    for i in range(d):
        sum1 += x[i] ** 2
        sum2 += np.cos(c * x[i])
    sum1 = -0.2 * np.sqrt(sum1/d)
    sum2 /= d
    f = np.exp(1) - 20.0 * np.exp(sum1) - np.exp(sum2) + 20.0

    return f


"""6. Weierstrass Function"""


"""7. Griewank’s Function"""


def rastrigin_func(x):
    """8. Rastrigin’s Function"""
    f = 0
    for i in range(len(x)):
        f += x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i]) + 10
    return f


def schwefel_func(x):
    """9. Modified Schwefel’s Function"""
    sum = 0
    d = len(x)
    for i in range(d):
        x[i] += 4.209687462275036e+002

        if x[i] > 500:
            tmp1 = x[i] % 500
            sum += (500 - tmp1) * np.sin(np.power(500 - tmp1, 0.5))
            tmp2 = (x[i] - 500) / 100
            sum -= tmp2 ** 2 / d
        elif x[i] < -500:
            tmp1 = np.abs(x[i]) % 500
            sum += (tmp1 - 500) * np.sin(np.power(500 - tmp1, 0.5))
            tmp2 = (x[i] + 500) / 100
            sum -= tmp2 ** 2 / d
        else:
            sum += x[i] * np.sin(np.power(np.abs(x[i]), 0.5))

    f = 4.189828872724338e+002 * d - sum
    return f


"""10. Katsuura Function"""


"""11. HappyCat Function"""


"""12. HGBat Function"""


"""13. Expanded Griewank’s plus Rosenbrock’s Function"""


"""14. Expanded Scaffer’s F6 Function"""
