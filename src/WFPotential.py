import numpy as np


def zero(x):
    V = np.zeros(x.shape)
    return V


def linear(x, m=0.005):
    V = np.zeros(x.shape)
    for x_index, x_value in enumerate(x):
        if x_value >= 0:
            V[x_index] = m * x_value
    return V


def wall(x, V_0=0.5, a=3.):
    V = np.zeros(x.shape)
    for x_index, x_value in enumerate(x):
        if 0 < x_value < a:
            V[x_index] = V_0
    return V


def step(x, V_0=1.):
    return wall(x, V_0, 10.)


def quadratic(x, omega=2.0*np.pi):
    V = np.zeros(x.shape)
    for x_index, x_value in enumerate(x):
        V[x_index] = 0.00001 * omega * x_value ** 2
    return V
