import numpy as np
from lib import WFGeneral as wg

E = wg.E
m = wg.m

# dimensionless wave numbers k_0 and kappa_0
p0 = wg.p_0
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

# Computation of GAUSS-HERMITE abscissas and weights.
x_H, w = np.polynomial.hermite.hermgauss(200)

k_b = wg.kb_a
k_ = np.sqrt(E) * k_b


def phi_alpha(x, t, p=0):
    """Stationary solution (scattering by a rectangle-barrier) superposed with psi_t."""
    psi_xt = np.zeros(x.size, complex)
    p_0 = 2 * wg.sigma_p * p + p0
    if p_0 ** 2 <= m:
        k_0 = np.sqrt(m - p_0 ** 2)
    else:
        k_0 = 1j * np.sqrt(p_0 ** 2 - m)
    t = t - wg.t_col(p_0)

    for i in range(x.size):
        if x[i] < -1.:
            psi_xt[i] = (A * np.exp(1j * p_0 * x[i]) +
                         B * np.exp(-1j * p_0 * x[i]))
        elif -1. <= x[i] <= 1.:
            psi_xt[i] = (C * np.exp(-k_0 * x[i]) +
                         D * np.exp(k_0 * x[i]))
        elif x[i] > 1.:
            psi_xt[i] = F * np.exp(1j * p_0 * x[i])
    return psi_xt * wg.psi_t(t, p_0)


def psi(x, t, phi=phi_alpha):
    """Approximation of wavepacket by GAUSS-HERMITE procedure."""
    psi = 0
    for j in range(0, len(x_H), 1):
        psi += w[j] * phi(x, t, x_H[j])
    return psi


def probs(psi):
    """Scattering probabilities."""
    print('\nAnalytical\n'
          f'Reflection probability: {round(np.abs(B / A) ** 2, 4)}\n'
          f'Transmission probability: {round(np.abs(F / A) ** 2, 4)}')
    print('\nGAUSS-HERMITE')
    wg.probs(psi)


def x_t(t):
    """
    Position of classical particle x(t).
    :param t: time coord
    :return: position
    """
    x_0 = wg.x_0
    x_pos = x_0 + (2. * p0 / m) * t
    if x_pos >= -1.:
        t_col = (-1. - x_0) / (2. * p0 / m)
        return -1. - (2. * p0 / m) * (t - t_col)
    else:
        return x_pos
