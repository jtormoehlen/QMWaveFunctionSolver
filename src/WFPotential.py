import numpy as np


def linear(x, m=400.):
    V = np.zeros(x.shape)
    for x_index, x_value in enumerate(x):
        if x_value >= 0:
            V[x_index] = m * x_value
    return V


def wall(x, x_0=0., V_0=1000., a=0.5):
    V = np.zeros(x.shape)
    for x_index, x_value in enumerate(x):
        if x_0 < x_value < x_0 + a:
            V[x_index] = V_0
    return V


def step(x, x_0=0., V_0=1000.):
    return wall(x, x_0, V_0, 5.)
