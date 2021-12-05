import numpy as np
import WaveFunction as wf

p0 = np.sqrt(2. * wf.m_e * wf.E_e) / wf.hbar
k0 = np.sqrt(2. * wf.m_e * (wf.V_b - wf.E_e)) / wf.hbar

# Coefficients of stationary solution (scattering on a barrier) for all three regions
A = 1.0
F = A / ((np.cosh(2 * k0 * wf.a_b) + (1j / 2) * (k0 / p0 - p0 / k0) * np.sinh(2 * k0 * wf.a_b)) * np.exp(2j * p0 * wf.a_b))
B = F * (-1j / 2) * (k0 / p0 + p0 / k0) * np.sinh(2 * k0 * wf.a_b)
C = (F / 2) * (1 - (1j * p0 / k0)) * np.exp((k0 + 1j * p0) * wf.a_b)
D = (F / 2) * (1 + (1j * p0 / k0)) * np.exp((-k0 + 1j * p0) * wf.a_b)

# Computation of GAUSS-HERMITE abscissas and weights with orthonormal set of polynomials.
x_H, w = np.polynomial.hermite.hermgauss(300)


def psi_x(x, t, p=0):
    """Stationary solution (scattering by a rectangle-barrier) superpositioned with psi_t."""
    psi_xt = np.zeros(x.size, complex)
    p_0 = 2 * wf.sigma_p * p + wf.p_
    if p_0 ** 2 <= 2. * wf.m_e * wf.V_b:
        k_0 = np.sqrt(2. * wf.m_e * wf.V_b - p_0 ** 2)
    else:
        k_0 = 0.
    t = t - 2. * wf.t_col(p_0)

    for i in range(x.size):
        if x[i] < -wf.a_b:
            psi_xt[i] = (A * np.exp(1j * p_0 / wf.hbar * x[i]) +
                         B * np.exp(-1j * p_0 / wf.hbar * x[i]))
        elif -wf.a_b <= x[i] <= wf.a_b:
            psi_xt[i] = (C * np.exp(-k_0 / wf.hbar * x[i]) +
                         D * np.exp(k_0 / wf.hbar * x[i]))
        elif x[i] > wf.a_b:
            psi_xt[i] = F * np.exp(1j * p_0 / wf.hbar * x[i])
    return wf.psi0_norm * psi_xt * wf.psi_t(t, p_0)


def psi(x, t, phi):
    """Approximation of wavepacket by GAUSSIAN quadrature procedure"""
    F = 0.0
    for j in range(0, len(x_H), 1):
        F += w[j] * phi(x, t, x_H[j])
    return F


def info(psi):
    """Scattering probabilities for the console."""
    print('\nAnalytical\n'
          f'Reflection probability: {round(np.abs(B / A) ** 2, 4)}\n'
          f'Transmission probability: {round(np.abs(F / A) ** 2, 4)}')
    print('\nGAUSS-QUAD')
    wf.prob_info(psi)


def x_t(t):
    x_pos = wf.x_0 + (wf.p_ / wf.m_e) * t
    if x_pos >= -wf.a_b:
        t_col = (-wf.a_b - wf.x_0) / (wf.p_ / wf.m_e)
        return -wf.a_b - (wf.p_ / wf.m_e) * (t - t_col)
    else:
        return x_pos
