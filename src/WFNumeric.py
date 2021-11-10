import numpy as np
import scipy as sp
from scipy import integrate
from scipy.sparse import linalg as ln
import WaveFunction as wf

x = wf.x
dx = wf.dx
V = wf.V(x)

t_0 = 0.0
t_n = wf.t_col()
dt = t_n / 100
t_N = np.arange(t_0, t_n, dt)


class PsiIVP:
    def psi_ivp(self):
        return integrate.solve_ivp(self.psi_0, y0=wf.psi(x, 0, wf.free),
                                   t_span=[t_0, t_n], t_eval=t_N, method='RK23')

    def psi_0(self, t, psi):
        D = sp.sparse.diags([1, -2, 1],
                            [-1, 0, 1],
                            shape=(x.size, x.size)) / dx ** 2
        return 1j * (1 / wf.m * D.dot(psi) - V * psi)


class PsiCN:
    h_diag = sp.ones(x.size) / dx ** 2 - V
    h_non_diag = sp.ones(x.size - 1) * (-0.5 / dx ** 2)
    hamiltonian = sp.sparse.diags([h_diag, h_non_diag, h_non_diag], [0, 1, -1])

    implicit = (sp.sparse.eye(x.size) - dt / (1j * wf.m) * hamiltonian).tocsc()
    explicit = (sp.sparse.eye(x.size) + dt / (1j * wf.m) * hamiltonian).tocsc()
    evolution_matrix = ln.inv(implicit).dot(explicit).tocsr()

    def __init__(self):
        self.psi = wf.psi(x, 0, wf.free)

    def step(self, t):
        if t > 0:
            self.psi = self.evolution_matrix.dot(self.psi)

    def get(self):
        return self.psi
