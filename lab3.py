# Optimizacijos Metodai Lab3
# Tomas Kvaraciejus, 2110588

import math
import numpy as np
import numdifftools as nd

class points:
    x: float
    y: float
    z: float

    def __init__(self, _x, _y, _z):
        self.x = _x
        self.y = _y
        self.z = _z

    def deduct(self, x,y,z):
        self.x -= x
        self.y -= y
        self.z -= z

precision = 10 ** -7
precision_bauda = 10 ** -4
def unwrap_to_points(xyz):
    return points(xyz[0], xyz[1], xyz[2])
def func(p: points):
    return -(p.x * p.y * p.z)
def func_from_arr(xyz):
    return func(unwrap_to_points(xyz))
def h_func(xyz):
    p = unwrap_to_points(xyz)
    return 1 - (2 * p.x * p.y) - (2 * p.x * p.z) - (2 * p.y * p.z)

starting_points = [
    points(0.0, 0.0, 0.0),
    points(1.0, 1.0, 1.0),
    points(0.5, 0.8, 0.8)
    ]

bandomos_baudos = [
    (0.5, 1.2),
    (0.75, 1.5),
    (1.0, 2.0)
]

def greic_nusileidimas(curr_func, p: points):
    f_old = math.inf
    f_new = curr_func([p.x, p.y, p.z])

    while precision < abs(f_old - f_new):
        grad_x, grad_y, grad_z = nd.Gradient(curr_func)([p.x, p.y, p.z])
        if abs(grad_x) <= 10 ** -6:
            grad_x -= 10 ** -6
        if abs(grad_y) <= 10 ** -6:
            grad_y -= 10 ** -6
        if abs(grad_z) <= 10 ** -6:
            grad_z -= 10 ** -6

        gamma_values = np.linspace(0, 100, num=10000)
        step_size = 0

        f_min = math.inf
        f_value = math.inf
        for gamma in gamma_values:
            f_old = f_value
            f_value = curr_func([p.x - gamma * grad_x, p.y - gamma * grad_y, p.z - gamma * grad_z])
            if f_value < f_min:
                f_min = f_value
                step_size = gamma
            if f_old < f_value:
                break

        p.deduct(step_size * grad_x, step_size * grad_y, step_size * grad_z)
        f_old = f_new
        f_new = curr_func([p.x, p.y, p.z])

    return p

def geriausia_bauda(p, bauda, bauda_mult):
    f_old = math.inf
    f_new = func(p)
    i = 0

    while precision_bauda < abs(f_old - f_new):
        curr_func = lambda xyz: (func_from_arr(xyz) + bauda *
                                 ((min(0, xyz[0]) ** 2 + min(0, xyz[1]) ** 2 + min(0, xyz[2]) ** 2) + h_func(xyz) ** 2))
        p = greic_nusileidimas(curr_func, p)
        bauda *= bauda_mult
        f_old = f_new
        f_new = curr_func([p.x, p.y, p.z])
        i += 1
    
    return p, i

class lab3:
    for p in starting_points:
        print("pradiniai taskai: [{:1}, {:1}, {:1}]\n".format(p.x, p.y, p.z))
        for bauda, baudos_daugiklis in bandomos_baudos:
            print("pradine bauda: {:1}".format(bauda), ", baudos daugiklis: {:1}".format(baudos_daugiklis))
            p_copy = points(p.x, p.y, p.z)
            (taskai, iteracijos) = geriausia_bauda(p_copy, bauda, baudos_daugiklis)
            print("gauti taskai: [{:.12f}, {:.12f}, {:.12f}], gautas f-jos rezultatas: {:.12f}, iteraciju kiekis: {}\n".format(taskai.x, taskai.y, taskai.z, func(taskai), iteracijos))
            
        print("--------------------\n")