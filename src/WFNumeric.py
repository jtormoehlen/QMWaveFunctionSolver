import numpy as np
import scipy as sp
from scipy import integrate
from scipy.sparse import linalg as ln
import WaveFunction as wf
import WFAnalytic as wfa

x = wf.x
dx = wf.dx
V = wf.V(x)

# Discretization of time coords from zero to collision time
t_0 = 0.0
t_n = wf.t_col()
dt = t_n / 100
t_N = np.arange(t_0, t_n, dt)


def hamilton():
    """Discretization of HAMILTON-operator by finite difference method."""
    H_diag = 2 * sp.ones(x.size) / (wf.m * dx ** 2) + V
    H_non_diag = sp.ones(x.size - 1) * (-1 / (wf.m * dx ** 2))
    return sp.sparse.diags([H_non_diag, H_diag, H_non_diag], [-1, 0, 1])


class PsiRK:
    def solve(self):
        """Solve schrodinger-equation by RUNGE-KUTTA procedure for initial free wavepacket."""
        return integrate.solve_ivp(self.eq, y0=wfa.psi(x, 0, wfa.psi_x),
                                   t_span=[t_0, t_n], t_eval=t_N, method='RK23')

    @staticmethod
    def eq(t, psi):
        """Right-hand side of schrodinger-equation."""
        H = hamilton()
        return 1j * -H.dot(psi)

    @staticmethod
    def info(psi):
        norm = wf.norm(psi)
        refl = wf.prob(np.abs(psi) ** 2 / norm, -wf.x_max, -wf.a)
        trans = wf.prob(np.abs(psi) ** 2 / norm, wf.a, wf.x_max)
        print('RUNGE-KUTTA\n'
              f'Reflection probability: {round(refl, 4)}\n'
              f'Transmission probability: {round(trans, 4)}')


class PsiCN:
    """Discretization of position and time coords by CRANK-NICOLSON procedure."""
    H = hamilton()
    implicit = (sp.sparse.eye(x.size) - dt / 2j * H).tocsc()
    explicit = (sp.sparse.eye(x.size) + dt / 2j * H).tocsc()
    evolution_matrix = ln.inv(implicit).dot(explicit).tocsr()

    def __init__(self):
        """Initiate free wavepacket psi0."""
        self.psi = wfa.psi(x, 0, wfa.psi_x)

    def evolve(self, t):
        """Solve schrodinger-equation by applying CN-matrix."""
        if 0 < t < t_N.size:
            self.psi = self.evolution_matrix.dot(self.psi)
        else:
            self.psi = wfa.psi(x, 0, wfa.psi_x)
        return self.psi

    def custom_evolve(self, t_end):
        """CN-matrix with custom time step Delta t applied to psi0."""
        psi0 = wfa.psi(x, 0, wfa.psi_x)
        delta_t = 0.0
        while delta_t <= t_end:
            psi0 = self.evolution_matrix.dot(psi0)
            delta_t += dt
        return psi0

    def info(self):
        """Some information about scattering probabilities for the console."""
        psi = self.custom_evolve(t_n)
        norm = wf.norm(psi)
        refl = wf.prob(np.abs(psi) ** 2 / norm, -wf.x_max, -wf.a)
        trans = wf.prob(np.abs(psi) ** 2 / norm, wf.a, wf.x_max)
        print('CRANK-NICOLSON\n'
              f'Reflection probability: {round(refl, 4)}\n'
              f'Transmission probability: {round(trans, 4)}')
