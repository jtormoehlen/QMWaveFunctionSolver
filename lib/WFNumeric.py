import numpy as np
import scipy as sp
from scipy import integrate
from scipy.sparse import linalg as ln
from lib import WaveFunction as wf

x = wf.x_j
dx = wf.dx
V = wf.V(x)
m = wf.m
psi_0 = wf.psi_0(x)

# discrete time coords (0,dt,2*dt,...,t_0)
t_N = wf.t_0()
dt = t_N / 100
t_n = np.arange(0.0, t_N, dt)

# Discrete HAMILTON-operator
def hamilton():
    d0 = sp.ones(x.size) * 2. / (m * dx ** 2) + V
    d1 = sp.ones(x.size - 1) * -1. / (m * dx ** 2)
    return sp.sparse.diags([d1, d0, d1], [-1, 0, 1])

class RKSolver:
    # Solution of schrodinger-equation by RUNGE-KUTTA procedure for initial free wave packet
    def __init__(self):
        self.__sol = integrate.solve_ivp(self.__dt_psi, y0=psi_0,
                                         t_span=[min(t_n), max(t_n)], t_eval=t_n, method='RK23')
        self.probs()

    # Right-hand side of schrodinger-equation
    @staticmethod
    def __dt_psi(t, psi):
        H = hamilton()
        return -1j * H.dot(psi)

    # Solution psi(x,t)
    # :param t: time index in (0,1,...,|t_n|)
    # :return: wave function
    def psi(self, t):
        return self.__sol.y[:, t]

    # Scattering probabilities
    def probs(self):
        print('\nRUNGE-KUTTA')
        wf.probs(self.psi(-1))


class CNSolver:
    # Solution of schrodinger-equation by CRANK-NICOLSON procedure for initial free wave packet
    __H = hamilton()
    __imp = (sp.sparse.eye(x.size) - dt / 2j * __H).tocsc()
    __exp = (sp.sparse.eye(x.size) + dt / 2j * __H).tocsc()
    __evol_mat = ln.inv(__imp).dot(__exp).tocsr()

    # Initial free wave packet psi0
    def __init__(self):
        self.__sol = [psi_0]
        for t in range(t_n.size):
            self.__sol.append(self.__evolve())
        self.probs()

    # Evolve psi^n_j to psi^n_j+1
    def __evolve(self):
        return self.__evol_mat.dot(self.__sol[-1])

    # Solution psi(x,t)
    # :param t: time index in (0,1,...,|t_n|)
    # :return: wave function
    def psi(self, t):
        return self.__sol[t]

    # Scattering probabilities
    def probs(self):
        print('\nCRANK-NICOLSON')
        wf.probs(self.psi(-1))
