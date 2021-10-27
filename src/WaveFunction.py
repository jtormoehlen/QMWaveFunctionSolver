import numpy as np
import WFGaussQuad as gq


a = 1.0
x_0 = -50
V_0 = 1.0
E_0 = 0.95 * V_0

k = np.sqrt(2 * E_0)
sigma_E = E_0 / 100
sigma_p = np.sqrt(2 * sigma_E)
norm = 1 / ((2 * np.pi) ** 0.25 * np.sqrt(sigma_p))

kappa = np.sqrt(2 * (V_0 - E_0))
epsilon = kappa / k - k / kappa
eta = kappa / k + k / kappa
A = 1.0
F = A / ((np.cosh(2 * kappa * a) + (1j * epsilon * np.sinh(2 * kappa * a)) / 2) * np.exp(2j * k * a))
B = F * ((-1j * eta * np.sinh(2 * kappa * a)) / 2)
C = ((1 - 1j * k / kappa) * np.exp((kappa + 1j * k) * a) * F) / 2
D = ((1 + 1j * k / kappa) * np.exp((-kappa + 1j * k) * a) * F) / 2


def free(x, t, p=0):
    k_0 = 2.0 * sigma_p * p + k
    return norm * np.exp(1j * k_0 * (x - x_0)) * np.exp(-1j * (k_0 ** 2 / 2) * t)


def wall(x, t, p=0):
    psi = np.zeros(x.size, complex)
    k_0 = 2.0 * sigma_p * p + k
    kappa_0 = 2.0 * sigma_p * p + np.sqrt(2 * (V_0 - E_0))
    t_col = (-x_0 - a) / k
    for i in range(x.size):
        x_i = x[i] - x_0
        if x[i] < -a:
            psi[i] = (A * np.exp(1j * k_0 * (x[i] - x_0)) + B * np.exp(-1j * k_0 * (x[i] + x_0))) * time(t, k_0)
        elif -a <= x[i] <= a:
            if t >= t_col:
                psi[i] = (C * np.exp(-kappa_0 * (x[i] - a)) + D * np.exp(kappa_0 * (x[i] + a))) * time(t - t_col, k_0)
        else:
            psi[i] = F * np.exp(1j * k_0 * (x[i] - x_0)) * time(t, k_0)
    return psi * norm


def sigma_x(t):
    return np.sqrt((1 / (2 * sigma_p)) ** 2 + sigma_p ** 2 * t ** 2)


def time(t, p):
    return np.exp(-1j * (p ** 2 / 2) * t)


def psi(x, t):
    return gq.gauss_quad(x, t, wall)


def V(x):
    V_N = np.zeros(x.size)
    for i, v in enumerate(x):
        if -a <= v <= a:
            V_N[i] = V_0
    return V_N


def info():
    print('################################')
    print('Energy level: ' + str(E_0 / V_0) + ' V_0')
    print('Initial position: ' + str(x_0) + ' a')
    print('Barrier width: ' + str(2 * a) + ' a')
    print('Transmission probability: ' + str(round(np.abs(F / A) ** 2, 4)))
    print('Reflection probability: ' + str(round(np.abs(B / A) ** 2, 4)))
    print('################################')
