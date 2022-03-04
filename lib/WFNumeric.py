import numpy as np
import scipy as sp
from scipy import integrate
from scipy.sparse import linalg as ln
from lib import WFGeneral as wg

x = wg.x_j
dx = wg.dx
V = wg.V(x)
m = wg.m
psi_0 = wg.psi_0(x)

# discrete time coords (t_0,t_0+dt,...,t_col)
t_0 = 0.0
t_N = wg.t_col()
dt = t_N / 100
t_n = np.arange(t_0, t_N, dt)


def hamilton():
    """Discrete HAMILTON-operator."""
    d0 = sp.ones(x.size) * 2. / (m * dx ** 2) + V
    d1 = sp.ones(x.size - 1) * -1. / (m * dx ** 2)
    return sp.sparse.diags([d1, d0, d1], [-1, 0, 1])


class RKSolver:
    """Solution of schrodinger-equation by RUNGE-KUTTA procedure for initial free wave packet."""
    def __init__(self):
        self.sol = integrate.solve_ivp(self.dt_psi, y0=psi_0,
                                       t_span=[t_0, t_N], t_eval=t_n, method='RK23')

    @staticmethod
    def dt_psi(t, psi):
        """Right-hand side of schrodinger-equation."""
        H = hamilton()
        return -1j * H.dot(psi)

    def psi(self, t):
        """
        Solution psi(x,t).
        :param t: time index in (0,1,...,|t_n|)
        :return: wave function
        """
        return self.sol.y[:, t]

    def prob_info(self):
        """Scattering probabilities."""
        print('\nRUNGE-KUTTA')
        wg.prob_info(self.psi(-1))


class CNSolver:
    """Solution of schrodinger-equation by CRANK-NICOLSON procedure for initial free wave packet."""
    H = hamilton()
    imp = (sp.sparse.eye(x.size) - dt / 2j * H).tocsc()
    exp = (sp.sparse.eye(x.size) + dt / 2j * H).tocsc()
    evol_mat = ln.inv(imp).dot(exp).tocsr()

    def __init__(self):
        """Initial free wave packet psi0."""
        self.sol = [psi_0]
        for t in range(t_n.size):
            self.sol.append(self.evolve())

    def evolve(self):
        """Evolve psi^n_j to psi^n_j+1."""
        return self.evol_mat.dot(self.sol[-1])

    def psi(self, t):
        """
        Solution psi(x,t).
        :param t: time index in (0,1,...,|t_n|)
        :return: wave function
        """
        return self.sol[t]

    def prob_info(self):
        """Scattering probabilities."""
        print('\nCRANK-NICOLSON')
        wg.prob_info(self.psi(-1))
