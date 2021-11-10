import numpy as np

import WFGaussQuad as gq

x_max = 50
n_x = 201
x, dx = np.linspace(-x_max, x_max, n_x, retstep=True)

a = 1.0
x_0 = -25
V_0 = 1.0
E_0 = 0.99
m = 0.5

sigma_E = E_0 / 100
sigma_p = np.sqrt(m * sigma_E)
sigma_x = 1 / (2 * sigma_p)
# norm = 1 / ((2 * np.pi) ** 0.25 * np.sqrt(sigma_p))

p_ = np.sqrt(m * E_0)
kappa_ = np.sqrt(m * (1.0 - E_0))


def free(x, t, p=0):
    p_0 = 2.0 * sigma_p * p + p_
    return np.exp(1j * p_0 * (x - x_0)) * psi_t(t, p_0)


def psi_x(x, t, p=0):
    psi = np.zeros(x.size, complex)
    p_0 = 2 * sigma_p * p + p_
    kappa_0 = 2 * sigma_p * p + kappa_
    t = t - t_col(p_0)
    A, B, C, D, F = psi_coeffs(p_0, kappa_0)
    for i in range(x.size):
        if x[i] < -a:
            psi[i] = (A * np.exp(1j * p_0 * x[i]) +
                      B * np.exp(-1j * p_0 * x[i]))
        elif -a <= x[i] <= a:
            psi[i] = (C * np.exp(-kappa_0 * x[i]) +
                      D * np.exp(kappa_0 * x[i]))
        elif x[i] > a:
            psi[i] = F * np.exp(1j * p_0 * x[i])
    psi = psi * psi_t(t, p_0)
    return psi


def psi_t(t, p):
    return np.exp(-1j * (p ** 2 / m) * t)


def psi(x, t, f=psi_x):
    return gq.gauss_quad(x, t, f)


def V(x):
    V_N = np.zeros(x.size)
    for i, v in enumerate(x):
        if -a <= v <= a:
            V_N[i] = V_0
    return V_N


def psi_coeffs(p=p_, kappa=kappa_):
    A = 1.0
    F = A / ((np.cosh(2 * kappa) + (1j / 2) * (kappa / p - p / kappa) * np.sinh(2 * kappa)) * np.exp(2j * p))
    B = F * (-1j / 2) * (kappa / p + p / kappa) * np.sinh(2 * kappa)
    C = (F / 2) * (1 - (1j * p / kappa)) * np.exp(kappa + 1j * p)
    D = (F / 2) * (1 + (1j * p / kappa)) * np.exp(-kappa + 1j * p)
    return [A, B, C, D, F]


def t_col(p):
    return (-a - x_0) / (p / m)


def info():
    A, B, C, D, F = psi_coeffs()
    print('################################')
    print('Energy level: ' + str(E_0 / V_0) + ' V_0')
    print('Initial position: ' + str(x_0) + ' a')
    print('Barrier width: ' + str(2 * a) + ' a')
    print('Transmission probability: ' + str(round(np.abs(F / A) ** 2, 4)))
    print('Reflection probability: ' + str(round(np.abs(B / A) ** 2, 4)))
    print('################################')
