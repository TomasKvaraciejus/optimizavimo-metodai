# Optimizacijos Metodai Lab1
# Tomas Kvaraciejus, 2110588
# f(x) = (x^2−8)^2/8 - 1

import math
import numpy as np
from autograd import grad
from scipy import constants
import matplotlib.pyplot as plt

precision = 10 ** -4
initL = 0
initR = 10
x0 = 5 

# tikslo funkcija
def f(x):
    return (((x ** 2 - 8) ** 2) / 8) - 1

# vizualizavimo funkcija
def visualize(minPoints, minValues, name = "graph"):
    xPoints = np.linspace(initL, initR, 50)
    yPoints = f(xPoints)

    plt.ylim(-100, 1000)
    plt.xlim(-0.5, 10.5)
    plt.plot(xPoints, yPoints, "black")
    plt.title(name)
    plt.xlabel('x', color='#1C2833')
    plt.ylabel('f(x)', color='#1C2833')
    plt.grid()

    for x, y in zip(minPoints, minValues):
        plt.plot(x, y, marker="x", markersize=5, markeredgecolor="red")

    plt.show()

# dalijimo pusiau metodas
def m_dp():
    l = initL
    r = initR
    L = 1   
    xm = (l + r) / 2
    fxm = f(xm)
    minPoints = list()
    minValues = list()

    while L > precision:
        L = r - l

        x1 = l + L/4
        x2 = r - L/4

        fx1 = f(x1)
        fx2 = f(x2)

        if fx1 < fxm:
            r = xm
            xm = x1
            fxm = fx1

        elif fx2 < fxm:
            l = xm
            xm = x2
            fxm = fx2

        else:
            l = x1
            r = x2

        minPoints.append(xm)
        minValues.append(fxm)

    return minPoints, minValues

# auksinio pjuvio metodas
def m_gr():
    l = initL
    r = initR
    L = r - l
    minPoints = list()
    minValues = list()

    x1 = r - (L / constants.golden)
    x2 = l + (L / constants.golden)

    while L > precision: 
        fx1 = f(x1)
        fx2 = f(x2)

        if fx1 > fx2:
            xm = x2
            fxm = fx2

            l = x1
            x1 = x2
            L = r - l
            x2 = l + (L / constants.golden)

        else:
            xm = x1
            fxm = fx1

            r = x2
            x2 = x1
            L = r - l
            x1 = r - (L / constants.golden)
    
        minPoints.append(xm)
        minValues.append(fxm)

    return minPoints, minValues

# Niutono metodas
def m_n():
    minPoints = list()
    minValues = list()

    x = float(x0)

    df = grad(f)
    ddf = grad(df)

    L = math.inf

    while abs(L) > precision:
        L = df(x) / ddf(x)
        x = x - L
        fx = f(x)

        minPoints.append(x)
        minValues.append(fx)

    return minPoints, minValues

class lab1:
    print('metodas | iskvietimai | xm | f(xm)')

    minPoints, minValues = m_dp()
    print("dalinimo pusiau metodas:", len(minPoints), minPoints[(len(minPoints) - 1)], minValues[(len(minValues) - 1)])
    visualize(minPoints, minValues)

    minPoints, minValues = m_gr()
    print("auksinio pjuvio metodas:", len(minPoints), minPoints[(len(minPoints) - 1)], minValues[(len(minValues) - 1)])
    visualize(minPoints, minValues)

    minPoints, minValues = m_n()
    print("Niutono metodas:", len(minPoints), minPoints[(len(minPoints) - 1)], minValues[(len(minValues) - 1)])
    visualize(minPoints, minValues)