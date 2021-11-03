import numpy as np
import WFGaussQuad as gq


a = 1.0
x_0 = -30
V_0 = 1.0
E_0 = 0.9 * V_0
m = 1.0

sigma_E = E_0 / 100
sigma_p = np.sqrt(2 * m * sigma_E)
norm = 1 / ((2 * np.pi) ** 0.25 * np.sqrt(sigma_p))

k = np.sqrt(2 * m * E_0)
kappa = np.sqrt(2 * m * (V_0 - E_0))
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


def psi_x(x, t, p=0):
    psi = np.zeros(x.size, complex)
    p_0 = 2.0 * sigma_p * p + k
    kappa_0 = 2.0 * sigma_p * p + np.sqrt(2 * (V_0 - E_0))
    t_col = (-x_0 - a) / k
    for i in range(x.size):
        if x[i] < -a:
            psi[i] = (A * np.exp(1j * p_0 * (x[i] - x_0)) +
                      B * np.exp(-1j * p_0 * (x[i] + x_0))) * psi_t(t, p_0)
        elif -a <= x[i] <= a:
            if t >= t_col:
                psi[i] = (C * np.exp(-kappa_0 * (x[i] - a)) +
                          D * np.exp(kappa_0 * (x[i] + a))) * psi_t(t - t_col, p_0)
        else:
            psi[i] = F * np.exp(1j * p_0 * (x[i] - x_0)) * psi_t(t, p_0)
    return psi


def psi_t(t, p):
    return np.exp(-1j * (p ** 2 / 2) * t)


def psi(x, t):
    return norm * gq.gauss_quad(x, t, psi_x)


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
