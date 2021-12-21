import numpy as np
import scipy as sp
from scipy import integrate
from scipy.sparse import linalg as ln
import WaveFunction as wf

x = wf.x
dx = wf.dx
V = wf.V(x)
m = wf.m
psi_0 = wf.psi_0(x)

# Discretization of time variable from zero to collision time
t_0 = 0.0
t_N = wf.t_col()
dt = t_N / 200
t_n = np.arange(t_0, t_N, dt)


def hamilton():
    """Discretization of HAMILTON-operator by finite difference method."""
    d0 = sp.ones(x.size) * 2. / (m * dx ** 2) + V
    d1 = sp.ones(x.size - 1) * -1. / (m * dx ** 2)
    return sp.sparse.diags([d1, d0, d1], [-1, 0, 1])


class RKSolver:
    """Solve schrodinger-equation by RUNGE-KUTTA procedure for initial free wave packet."""
    def __init__(self):
        self.sol = integrate.solve_ivp(self.dt_psi, y0=psi_0,
                                       t_span=[t_0, t_N], t_eval=t_n, method='RK23')

    @staticmethod
    def dt_psi(t, psi):
        """Right-hand side of schrodinger-equation."""
        H = hamilton()
        return -1j * H.dot(psi)

    def psi(self, index):
        return self.sol.y[:, index]

    def prob_info(self):
        """Scattering probabilities for the console."""
        print('\nRUNGE-KUTTA')
        wf.prob_info(self.psi(-1))


class CNSolver:
    """Discretization of position and time coords by CRANK-NICOLSON procedure."""
    H = hamilton()
    imp = (sp.sparse.eye(x.size) - dt / 2j * H).tocsc()
    exp = (sp.sparse.eye(x.size) + dt / 2j * H).tocsc()
    evolution_matrix = ln.inv(imp).dot(exp).tocsr()

    def __init__(self):
        """Initiate free wavepacket psi0."""
        self.psi = psi_0

    def evolve(self, t):
        """Solve schrodinger-equation by applying CN-matrix."""
        if 0 < t < t_n.size:
            self.psi = self.evolution_matrix.dot(self.psi)
        else:
            self.psi = psi_0
        return self.psi

    def custom_evolve(self, t_end):
        """CN-matrix with custom time step Delta t applied to psi0."""
        psi0 = psi_0
        delta_t = 0.0
        while delta_t <= t_end:
            psi0 = self.evolution_matrix.dot(psi0)
            delta_t += dt
        return psi0

    def prob_info(self):
        """Scattering probabilities for the console."""
        psi = self.custom_evolve(t_N)
        print('\nCRANK-NICOLSON')
        wf.prob_info(psi)
