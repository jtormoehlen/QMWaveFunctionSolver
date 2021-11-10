import numpy as np
from scipy import integrate, sparse

import WaveFunction as wf

x = wf.x
V = wf.V(x)


def psi_ivp_solve():
    # norm = 1.0 / (wf.sigma_x * np.sqrt(2 * np.pi))
    # psi_0 = np.sqrt(norm) * np.exp(-(x - wf.x_0) ** 2 / (4 * wf.sigma_x ** 2)) * np.exp(1j * wf.p_ * x)

    t_0 = 0.0
    t_n = 2 * wf.t_col(wf.p_)
    dt = t_n / 100
    t_N = np.arange(t_0, t_n, dt)
    return integrate.solve_ivp(psi_ivp_eq, t_span=[t_0, t_n], y0=wf.psi(x, 0, wf.free), t_eval=t_N, method='RK23')


def psi_ivp_eq(t, psi):
    hbar = 1.0
    dx = ((2 * x[x.size - 1]) / x.size)
    D = sparse.diags([1, -2, 1],
                     [-1, 0, 1],
                     shape=(x.size, x.size)) / dx ** 2
    return -1j * (-0.5 * hbar / wf.m * D.dot(psi) + V / hbar * psi)
