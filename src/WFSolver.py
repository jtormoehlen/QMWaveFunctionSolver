import numpy as np
from scipy import integrate
from scipy import sparse

import WFAnimation as wfa
import WFPotential as wfp

# spatial grid
dx = 0.005
x = np.arange(-5, 5, dx)

hbar = 1
m = 1
sigma = 0.5
x_0 = -2.5
E_0 = 900
k = np.sqrt(2. * m * E_0) / hbar

# initial wave packet
A = 1.0 / (sigma * np.sqrt(np.pi))  # normalization constant
psi0 = np.sqrt(A) * np.exp(-(x - x_0) ** 2 / (2.0 * sigma ** 2)) * np.exp(1j * k * x)
print('Total Probability: ', np.sum(np.abs(psi0)**2)*dx)

# discrete laplacian (finite difference)
D2 = sparse.diags([1, -2, 1],
                  [-1, 0, 1],
                  shape=(x.size, x.size)) / dx**2

# define potential V(x)
V = wfp.linear(x)


# define schrödinger equation
def psi_t(t, psi):
    return -1j * (- 0.5 * hbar / m * D2.dot(psi) + V / hbar * psi)


# solve initial value problem
dt = 0.001
t0 = 0.0
tf = 0.2
t_eval = np.arange(t0, tf, dt)

# solve schrödinger equation with runge-kutta
print('Solving initial value problem')
sol = integrate.solve_ivp(psi_t,
                          t_span=[t0, tf],
                          y0=psi0,
                          t_eval=t_eval,
                          method='RK23')

wfa.run(x, V, sol)
