import numpy as np
from lib import WaveFunction as wf

E = wf.E
m = wf.m

# dimensionless wave numbers k_0 and kappa_0
p0 = wf.p0
if E <= 1.:
    k0 = np.sqrt(m * (1. - E))
else:
    k0 = 1j * np.sqrt(m * (E - 1.))

# coefficients of stationary solution (scattering on a barrier)
A = 1.0
F = A / ((np.cosh(2 * k0) + (1j / 2) * (k0 / p0 - p0 / k0) * np.sinh(2 * k0)) * np.exp(2j * p0))
B = F * (-1j / 2) * (k0 / p0 + p0 / k0) * np.sinh(2 * k0)
C = (F / 2) * (1 - (1j * p0 / k0)) * np.exp(k0 + 1j * p0)
D = (F / 2) * (1 + (1j * p0 / k0)) * np.exp(-k0 + 1j * p0)

# Computation of GAUSS-HERMITE abscissas and weights
x_H, w = np.polynomial.hermite.hermgauss(200)

K = 2 * wf.sigma_p / (2 * np.pi * wf.sigma_p) ** 0.25

# Stationary solution (scattering by a rectangle-barrier) superposed with psi_t
def phi_alpha(x, t, p=p0):
    psi_xt = np.zeros(x.size, complex)
    if p ** 2 <= m:
        k = np.sqrt(m - p ** 2)
    else:
        k = 1j * np.sqrt(p ** 2 - m)
    t = t - wf.t_0(p)  # set time t->t-t_col

    for i in range(x.size):
        if x[i] < -1.:  # reflection region
            psi_xt[i] = (A * np.exp(1j * p * x[i]) +
                         B * np.exp(-1j * p * x[i]))
        elif -1. <= x[i] <= 1.:  # barrier region
            psi_xt[i] = (C * np.exp(-k * x[i]) +
                         D * np.exp(k * x[i]))
        elif x[i] > 1.:  # transmission region
            psi_xt[i] = F * np.exp(1j * p * x[i])
    return psi_xt * wf.psi_t(t, p)  # superpose time-dependent solution

# Approximation of wavepacket by GAUSS-HERMITE procedure
def psi(x, t, phi=phi_alpha):
    psi = 0
    for j in range(0, len(x_H), 1):
        p = 2 * wf.sigma_p * x_H[j] + p0  # momentum substitution for GH-proc
        psi += K * w[j] * phi(x, t, p)
    return psi

# Scattering probabilities
def probs(psi):
    print('\nAnalytical\n'
          f'Reflection probability: {round(np.abs(B / A) ** 2, 4)}\n'
          f'Transmission probability: {round(np.abs(F / A) ** 2, 4)}')
    print('\nGAUSS-HERMITE')
    wf.probs(psi)

# Position of classical particle x(t)
# :param t: time
# :return: position
def x_t(t):
    x_0 = wf.x0  # start pos
    x_pos = x_0 + (2. * p0 / m) * t  # pos at time t
    if x_pos >= -1.:
        t_col = (-1. - x_0) / (2. * p0 / m)
        return -1. - (2. * p0 / m) * (t - t_col)
    else:
        return x_pos

def v_x(x):
    E = p0 ** 2 / m
    V = wf.V(x)
    return 0.0