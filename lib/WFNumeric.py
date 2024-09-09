import numpy as np
import scipy as sp
from scipy import integrate
from scipy.sparse import linalg as ln
from lib.WaveFunction import Model
from lib import WaveFunction as wf

# x = wf.x_j
# dx = wf.dx
# V = wf.V(x)
# m = wf.m
# psi_0 = wf.psi_0(x)

# # discrete time coords (0,dt,2*dt,...,t_0)
# t_N = wf.t_0()
# dt = t_N / 100
# t_n = np.arange(0.0, t_N, dt)

# Discrete HAMILTON-operator
def H(x, dx, m, V):
    d0 = sp.ones(x.size)*2/(m*dx**2)+V(x)
    d1 = sp.ones(x.size-1)*-1/(m*dx**2)
    return sp.sparse.diags([d1, d0, d1], [-1, 0, 1])

# Right-hand side of schrodinger-equation
# @staticmethod
def dt_psi(t, psi):
    def H(x, dx, m, V):
        d0 = sp.ones(x.size)*2/(m*dx**2)+V(x)
        d1 = sp.ones(x.size-1)*-1/(m*dx**2)
        return sp.sparse.diags([d1, d0, d1], [-1, 0, 1])
    H = H(Model.x, Model.dx, Model.m, Model.V)
    return -1j*H.dot(psi)

class RKSolver(Model):
    
    # Solution of schrodinger-equation by RUNGE-KUTTA procedure for initial free wave packet
    def __init__(self, Model):
        self.sol = integrate.solve_ivp(dt_psi, y0=wf.psi_0(wf.x_j), t_span=[min(wf.t_n), max(wf.t_n)], t_eval=wf.t_n, method='RK23')
        self.probs()

    # Solution psi(x,t)
    # :param t: time index in (0,1,...,|t_n|)
    # :return: wave function
    def psi(self, t):
        return self.sol.y[:, t]

    # Scattering probabilities
    def probs(self):
        print('\nRUNGE-KUTTA')
        Model.probs(self.psi(-1))


class CNSolver(Model):
    # Solution of schrodinger-equation by CRANK-NICOLSON procedure for initial free wave packet
    H = H(Model.x, Model.dx, Model.m, Model.V)
    impMat = (sp.sparse.eye(Model.x.size)-Model.dt/2j*H).tocsc()
    expMat = (sp.sparse.eye(Model.x.size)+Model.dt/2j*H).tocsc()
    evolMat = ln.inv(impMat).dot(expMat).tocsr()

    # Initial free wave packet psi0
    def __init__(self, Model):
        self.sol = [wf.psi_0(wf.x_j)]
        for t in range(Model.t_n.size):
            self.sol.append(self.evolve())
        self.probs()

    # Evolve psi^n_j to psi^n_j+1
    def evolve(self):
        return self.evolMat.dot(self.sol[-1])

    # Solution psi(x,t)
    # :param t: time index in (0,1,...,|t_n|)
    # :return: wave function
    def psi(self, t):
        return self.sol[t]

    # Scattering probabilities
    def probs(self):
        print('\nCRANK-NICOLSON')
        Model.probs(self.psi(-1))
