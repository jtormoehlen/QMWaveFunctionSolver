import numpy as np
import WFUtil as wf

E = wf.E
m = wf.m

# Dimensionless wave numbers k_0 and kappa_0
p0 = wf.p_0
if E <= 1.:
    k0 = np.sqrt(m * (1. - E))
else:
    k0 = 1j * np.sqrt(m * (E - 1.))

# Coefficients of stationary solution (scattering on a barrier)
A = 1.0
F = A / ((np.cosh(2 * k0) + (1j / 2) * (k0 / p0 - p0 / k0) * np.sinh(2 * k0)) * np.exp(2j * p0))
B = F * (-1j / 2) * (k0 / p0 + p0 / k0) * np.sinh(2 * k0)
C = (F / 2) * (1 - (1j * p0 / k0)) * np.exp(k0 + 1j * p0)
D = (F / 2) * (1 + (1j * p0 / k0)) * np.exp(-k0 + 1j * p0)

# Computation of GAUSS-HERMITE abscissas and weights with orthonormal set of polynomials.
x_H, w = np.polynomial.hermite.hermgauss(300)

k_b = wf.kb_a
k_ = np.sqrt(E) * k_b
xi = 2. * k_ * wf.sigma_x ** 2 - 1j * (wf.x_0 + 1.)
F_0 = np.sqrt(2.) * (2. * np.pi * wf.sigma_x ** 2) ** 0.25 * np.exp(-wf.sigma_x ** 2 * k_ ** 2)


def phi_alpha(x, t, p=0):
    """Stationary solution (scattering by a rectangle-barrier) superpositioned with psi_t."""
    psi_xt = np.zeros(x.size, complex)
    p_0 = 2 * wf.sigma_p * p + p0
    if p_0 ** 2 <= m:
        k_0 = np.sqrt(m - p_0 ** 2)
    else:
        k_0 = 1j * np.sqrt(p_0 ** 2 - m)
    t = t - wf.t_col(p_0)

    for i in range(x.size):
        if x[i] < -1.:
            psi_xt[i] = (A * np.exp(1j * p_0 * x[i]) +
                         B * np.exp(-1j * p_0 * x[i]))
        elif -1. <= x[i] <= 1.:
            psi_xt[i] = (C * np.exp(-k_0 * x[i]) +
                         D * np.exp(k_0 * x[i]))
        elif x[i] > 1.:
            psi_xt[i] = F * np.exp(1j * p_0 * x[i])
    return psi_xt * wf.psi_t(t, p_0)


def psi(x, t, phi=phi_alpha):
    """Approximation of wavepacket by GAUSSIAN quadrature procedure."""
    psi = 0
    for j in range(0, len(x_H), 1):
        psi += w[j] * phi(x, t, x_H[j])
    return psi


def prob_info(psi):
    """Scattering probabilities for the console."""
    print('\nAnalytical\n'
          f'Reflection probability: {round(np.abs(B / A) ** 2, 4)}\n'
          f'Transmission probability: {round(np.abs(F / A) ** 2, 4)}')
    print('\nGAUSS-QUAD')
    wf.prob_info(psi)


def x_t(t):
    x_0 = wf.x_0
    x_pos = x_0 + (2. * p0 / m) * t
    if x_pos >= -1.:
        t_col = (-1. - x_0) / (2. * p0 / m)
        return -1. - (2. * p0 / m) * (t - t_col)
    else:
        return x_pos
