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
        self.__sol = integrate.solve_ivp(self.__dt_psi, y0=psi_0,
                                         t_span=[t_0, t_N], t_eval=t_n, method='RK23')
        self.probs()

    @staticmethod
    def __dt_psi(t, psi):
        """Right-hand side of schrodinger-equation."""
        H = hamilton()
        return -1j * H.dot(psi)

    def psi(self, t):
        """
        Solution psi(x,t).
        :param t: time index in (0,1,...,|t_n|)
        :return: wave function
        """
        return self.__sol.y[:, t]

    def probs(self):
        """Scattering probabilities."""
        print('\nRUNGE-KUTTA')
        wg.probs(self.psi(-1))


class CNSolver:
    """Solution of schrodinger-equation by CRANK-NICOLSON procedure for initial free wave packet."""
    __H = hamilton()
    __imp = (sp.sparse.eye(x.size) - dt / 2j * __H).tocsc()
    __exp = (sp.sparse.eye(x.size) + dt / 2j * __H).tocsc()
    __evol_mat = ln.inv(__imp).dot(__exp).tocsr()

    def __init__(self):
        """Initial free wave packet psi0."""
        self.__sol = [psi_0]
        for t in range(t_n.size):
            self.__sol.append(self.__evolve())
        self.probs()

    def __evolve(self):
        """Evolve psi^n_j to psi^n_j+1."""
        return self.__evol_mat.dot(self.__sol[-1])

    def psi(self, t):
        """
        Solution psi(x,t).
        :param t: time index in (0,1,...,|t_n|)
        :return: wave function
        """
        return self.__sol[t]

    def probs(self):
        """Scattering probabilities."""
        print('\nCRANK-NICOLSON')
        wg.probs(self.psi(-1))
