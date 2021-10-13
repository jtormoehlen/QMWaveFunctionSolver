import WFTestAnimation as wfa
import WFPotential as wfp

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.sparse import linalg as ln

E0 = 2.0
m = 1.0
k0 = np.sqrt(2.0 * m * E0)
x0 = -50.0

x_begin = -200.0
x_end = 200.0
dx = 0.5
x = np.arange(x_begin, x_end, dx)
dt = 1.0
t = np.arange(0, 200.0, dt)
sigma0 = 5.0
potential = wfp.zero(x)

norm = (2.0 * np.pi * sigma0 ** 2) ** (-0.25)
psi0 = np.exp(-(x - x0) ** 2 / (4.0 * sigma0 ** 2))
psi0 = psi0 * np.exp(1.0j * k0 * x)
# psi0 = psi0 * norm

o = np.ones((x.size), complex)
alp = (1j) * dt / (2 * dx ** 2) * o
xi = o + 1j * dt / 2 * (2 / (dx ** 2) * o + potential)
gam = o - 1j * dt / 2 * (2 / (dx ** 2) * o + potential)
diags = np.array([-1, 0, +1])
vecs1 = np.array([-alp, xi, -alp])
vecs2 = np.array([alp, gam, alp])
U1 = sp.sparse.spdiags(vecs1, diags, x.size, x.size)
U1 = U1.tocsc()
U2 = sp.sparse.spdiags(vecs2, diags, x.size, x.size)
U2 = U2.tocsc()

PSI_CN = np.zeros((x.size, t.size), complex)
PSI_CN[:, 0] = psi0
LU = sp.sparse.linalg.splu(U1)
for n in range(0, t.size - 1):
    b = U2.dot(PSI_CN[:, n])
    PSI_CN[:, n + 1] = LU.solve(b)

# wfa.run(x, t, PSI_CN)
wfa.show(x, 100, PSI_CN)
# wfa.show(x, 100, PSI_CN)
