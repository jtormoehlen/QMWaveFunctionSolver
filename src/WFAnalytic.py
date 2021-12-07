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

k_b = np.sqrt(2. * wf.m_e * wf.V_b) / wf.hbar
k__ = np.sqrt(wf.E_e / wf.V_b) * k_b
xi = 2. * k__ * wf.sigma_x ** 2 - 1j * (wf.x_0 + wf.a_b)
F_0 = np.sqrt(2.) * (2. * np.pi * wf.sigma_x ** 2) ** 0.25 * np.exp(-wf.sigma_x ** 2 * k__ ** 2)


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
    """Approximation of wavepacket by GAUSSIAN quadrature procedure."""
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


def kappa(k):
    if 0 <= k ** 2 <= k_b ** 2:
        return np.sqrt(k_b ** 2 - k ** 2)
    else:
        return 0.


def f_mu(mu, k):
    kappa0 = np.conj(kappa(k))
    if mu == 0:
        return np.cosh(kappa0 * wf.a_b) * np.cosh(k * xi) - \
                1j * kappa0 / k * np.sinh(kappa0 * wf.a_b) * np.sinh(k * xi)
    elif mu == 1:
        return -np.sinh(kappa0 * wf.a_b) * np.cosh(k * xi) + \
                1j * kappa0 / k * np.cosh(kappa0 * wf.a_b) * np.sinh(k * xi)


def C2_mu(mu, k):
    kappa0 = complex(kappa(k))
    if mu == 0:
        return 1. / np.pi * k ** 2 / (k ** 2 * np.cosh(kappa0 * wf.a_b) ** 2 +
                                      np.abs(kappa0) ** 2 * np.abs(np.sinh(kappa0 * wf.a_b)) ** 2)
    elif mu == 1:
        return 1. / np.pi * k ** 2 / (k ** 2 * np.abs(np.sinh(kappa0 * wf.a_b)) ** 2 +
                                      np.abs(kappa0) ** 2 * np.cosh(kappa0 * wf.a_b) ** 2)


def chi_mu(mu, x, k):
    kappa0 = complex(kappa(k))
    chi_alpha = np.zeros(x.size, complex)
    if mu == 0:
        for i in range(x.size):
            if x[i] < -wf.a_b:
                chi_alpha[i] = np.cosh(kappa0 * wf.a_b) * np.cos(k * (x[i] + wf.a_b)) - \
                               kappa0 / k * np.sinh(kappa0 * wf.a_b) * np.sin(k * (x[i] + wf.a_b))
            elif -wf.a_b <= x[i] <= wf.a_b:
                chi_alpha[i] = np.cosh(kappa0 * x[i])
            elif x[i] > wf.a_b:
                chi_alpha[i] = np.cosh(kappa0 * wf.a_b) * np.cos(k * (x[i] - wf.a_b)) + \
                               kappa0 / k * np.sinh(kappa0 * wf.a_b) * np.sin(k * (x[i] - wf.a_b))
        return np.sqrt(C2_mu(0, k)) * chi_alpha
    elif mu == 1:
        for i in range(x.size):
            if x[i] < -wf.a_b:
                chi_alpha[i] = -np.sinh(kappa0 * wf.a_b) * np.cos(k * (x[i] + wf.a_b)) + \
                                kappa0 / k * np.cosh(kappa0 * wf.a_b) * np.sin(k * (x[i] + wf.a_b))
            elif -wf.a_b <= x[i] <= wf.a_b:
                chi_alpha[i] = np.sinh(kappa0 * x[i])
            elif x[i] > wf.a_b:
                chi_alpha[i] = np.sinh(kappa0 * wf.a_b) * np.cos(k * (x[i] - wf.a_b)) + \
                               kappa0 / k * np.cosh(kappa0 * wf.a_b) * np.sin(k * (x[i] - wf.a_b))
        return np.sqrt(C2_mu(1, k)) * chi_alpha


def G_alpha(x, k):
    return np.heaviside(k, 1.) * F_0 / wf.sigma_x * \
           (C2_mu(0, k) * f_mu(0, k) * chi_mu(0, x, k) +
            C2_mu(1, k) * f_mu(1, k) * chi_mu(1, x, k))


def psi_alpha(x, t, k=0):
    psi_t = np.exp(-1j * k ** 2 * t / (2. * wf.m_e * wf.sigma_x ** 2 * wf.hbar))
    return psi_t * G_alpha(x, k / (wf.hbar * wf.sigma_x))
