# Optimizacijos Metodai Lab2
# Tomas Kvaraciejus, 2110588
# f(x) = ab(1 - (a + b))  / 8

import math
import numpy as np
import matplotlib.pyplot as plt

precision = 10 ** -4

# funkciju apibrezimai

def f(x, y):
    return -(((1 - x - y) * x * y) / 8)

def df_dx(x, y):
    return -((y / 8) - (y ** 2 / 8) - (x * y / 4))

def df_dy(x, y):
    return -((x / 8) - (x ** 2 / 8) - (x * y / 4))

def f_simplex(x, y):
    # guide towards valid values
    if x <= 0 or y <= 0:
        return 1 + abs(x) + abs(y)
    return -((1 - x - y) * x * y) / 8

# pavaizdavimo utilities

def plot(x_values, y_values):
    x_range = np.arange(-0.2, 1.4, 0.2)
    y_range = np.arange(-0.2, 1.4, 0.2)
    X, Y = np.meshgrid(x_range, y_range)
    Z = -f(X, Y)

    _, ax = plt.subplots()
    cs = ax.contour(X, Y, Z, levels=[-0.01, -0.005, -0.0025, 0, 1 / 300, 1 / 100, 1 / 50, 1 / 25, 1 / 15])
    ax.clabel(cs, fontsize=8, inline=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    length = len(x_values)
    current = 0
    for x, y in zip(x_values, y_values):
        ratio = current / length
        plt.plot(x, y, marker="o", markersize=5, markeredgecolor="black", markerfacecolor=(1 - ratio, ratio, 0.5))
        current += 1

    plt.show()

def run_comparisons(minimazimo_func):
    starting_points = [
        (0.001, 0.001),
        (0.999, 0.999),
        (0.8, 0.8)
        ]

    for x, y in starting_points:
        print(f"pradinis taskas: [{x}, {y}]")
        x_values, y_values = minimazimo_func(x, y)
        print(f"x: {x_values[-1]}, y: {y_values[-1]}, iteracijos: {len(x_values)}")
        plot(x_values, y_values)

# minimizavimo metodai

def grad_nusileidimas(x, y):
    x_values = [x]
    y_values = [y]
    step_size = 1

    while step_size > precision:
        grad_x = df_dx(x, y)
        grad_y = df_dy(x, y)
        step_size = math.sqrt(grad_x ** 2 + grad_y ** 2)

        x -= grad_x
        y -= grad_y

        x_values.append(x)
        y_values.append(y)

    return x_values, y_values

def greic_nusileidimas(x, y): 
    x_values = [x]
    y_values = [y]

    cur_value = math.inf
    prev_value = math.inf

    while (precision < abs(prev_value - cur_value)) or (prev_value == math.inf):
        print(prev_value - cur_value)
        grad_x = df_dx(x, y)
        grad_y = df_dy(x, y)

        gamma_values = np.logspace(np.log10(0.1), np.log10(100), num=1000)
        function_values = [f(x - gamma * grad_x, y - gamma * grad_y) for gamma in gamma_values]
        min = math.inf
        step_size = 0
        for i in range(1, len(function_values)):
            if min > function_values[i]:
                min = function_values[i]
                step_size = gamma_values[i]
            else:
                break

        x -= step_size * grad_x
        y -= step_size * grad_y

        x_values.append(x)
        y_values.append(y)

        prev_value = cur_value
        cur_value = f(x, y)

    return x_values, y_values

def simplex(x, y):
    simplex_points = [
        simplex_point(x, y),
        simplex_point(x + 0.9659258263, y + 0.2588190451),
        simplex_point(x + 0.2588190451, y + 0.9659258263)
    ]

    x_values = []
    y_values = []

    for _ in range(1000):
        x_values.append(simplex_points[0].x)
        y_values.append(simplex_points[0].y)

        sorted_simplex = sorted(simplex_points, key = lambda s: s.val)

        if (abs(sorted_simplex[1].x - sorted_simplex[0].x) + abs(sorted_simplex[1].y - sorted_simplex[0].y)) < precision:
            break

        point_C = simplex_point(
            (sorted_simplex[0].x + sorted_simplex[1].x) / 2,
            (sorted_simplex[0].y + sorted_simplex[1].y) / 2)

        point_R = simplex_point(
            2 * point_C.x - sorted_simplex[2].x,
            2 * point_C.y - sorted_simplex[2].y)

        if point_R.val < sorted_simplex[0].val:
            simplexE = simplex_point(
                point_C.x + 2 * (point_R.x - point_C.x),
                point_C.y + 2 * (point_R.y - point_C.y))
            
            if simplexE.val > point_R.val:
                sorted_simplex[2] = point_R
            else:
                sorted_simplex[2] = simplexE

        elif point_R.val < sorted_simplex[1].val:
            sorted_simplex[2] = point_R

        elif point_R.val < sorted_simplex[2].val:
            simplexE = simplex_point(
                point_C.x + 0.5 * (point_R.x - point_C.x),
                point_C.y + 0.5 * (point_R.y - point_C.y))

            if simplexE.val > point_R.val:
                sorted_simplex[2] = point_R
            else:
                sorted_simplex[2] = simplexE

        else:
            simplexE = simplex_point(
                point_C.x - 0.5 * (point_R.x - point_C.x),
                point_C.y + 0.5 * (point_R.y - point_C.y))
            
            if simplexE.val > point_R.val:
                sorted_simplex[2] = point_R
            else:
                sorted_simplex[2] = simplexE

        simplex_points = sorted_simplex

    return x_values, y_values

class simplex_point:
    x: float
    y: float
    val: float

    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y
        self.val = f_simplex(_x, _y)

    def __str__(self):
        return f"x: {self.x}, y: {self.y}, val: {self.val}"

class lab2:
    #print("gradientinis nusileidimas:")
    #run_comparisons(grad_nusileidimas)

    #print("greiciausias nusileidimas:")
    #run_comparisons(greic_nusileidimas)

    print("simplex:")
    run_comparisons(simplex)


