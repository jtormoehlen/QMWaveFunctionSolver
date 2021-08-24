import numpy as np
from scipy.optimize import fsolve
from scipy.special import factorial

# h = 6.62e-34
h = 1.
h_bar = h / (2. * np.pi)
# m_e = 9.11e-31
m_e = 1.


def energy_fin_pot(V_0, a, x0):
    kappa = np.sqrt(8. * m_e * V_0) * h_bar * a
    func = lambda x: (8. * x ** 4 - 8. * x ** 2 + 1) - np.cos(kappa * x)
    x_sol = fsolve(func, x0)
    E = V_0 * x_sol[0] ** 2
    return E


def hermite(n, x):
    if n == 0:
        return 1.
    elif n == 1:
        return 2 * x
    else:
        return 2 * x * hermite(n - 1, x) - 2 * (n - 1) * hermite(n - 2, x)


def hermite_const(n, omega):
    H_const = 1. / np.sqrt(2 ** n * factorial(n))
    y_const = np.sqrt((m_e * omega) / h_bar)
    return [H_const, y_const]
