import numpy as np
import WaveFunction as wf

p_ = wf.p_
k_ = wf.kappa_
a = wf.a

# Coefficients of stationary solution (scattering on a barrier) for all three regions
A = 1.0
F = A / ((np.cosh(2 * k_ * a) + (1j / 2) * (k_ / p_ - p_ / k_) * np.sinh(2 * k_ * a)) * np.exp(2j * p_ * a))
B = F * (-1j / 2) * (k_ / p_ + p_ / k_) * np.sinh(2 * k_ * a)
C = (F / 2) * (1 - (1j * p_ / k_)) * np.exp((k_ + 1j * p_) * a)
D = (F / 2) * (1 + (1j * p_ / k_)) * np.exp((-k_ + 1j * p_) * a)


def psi_x(x, t, p=0):
    """Stationary solution (scattering on a barrier) superpositioned with the time dependent solution."""
    psi_xt = np.zeros(x.size, complex)
    p_0 = 2 * wf.sigma_p * p + p_
    k_0 = 2 * wf.sigma_p * p + k_
    t = t - wf.t_col(p_0)

    for i in range(x.size):
        if x[i] < -wf.a:
            psi_xt[i] = (A * np.exp(1j * p_0 * x[i]) +
                         B * np.exp(-1j * p_0 * x[i]))
        elif -wf.a <= x[i] <= wf.a:
            psi_xt[i] = (C * np.exp(-k_0 * x[i]) +
                         D * np.exp(k_0 * x[i]))
        elif x[i] > wf.a:
            psi_xt[i] = F * np.exp(1j * p_0 * x[i])
    return psi_xt * wf.psi_t(t, p_0)


def gauss_hermite():
    """Computation of GAUSS-HERMITE abscissas and weights with orthonormal set of polynomials."""
    n = 100
    EPS = 1.0e-14
    x, w = np.zeros(n), np.zeros(n)
    m = int((n + 1) / 2)
    z = pp = 1.0

    for i in range(0, m, 1):
        if i == 0:
            z = np.sqrt((2 * n + 1) - 1.85575 * (2 * n + 1) ** -0.16667)
        elif i == 1:
            z -= 1.14 * (n ** 0.426) / z
        elif i == 2:
            z = 1.86 * z - 0.86 * x[0]
        elif i == 3:
            z = 1.91 * z - 0.91 * x[1]
        else:
            z = 2.0 * z - x[i - 2]

        for its in range(0, n, 1):
            p1 = 1 / np.pi ** 0.25
            p2 = 0.0

            for j in range(0, n, 1):
                p3 = p2
                p2 = p1
                p1 = z * np.sqrt(2.0 / (j + 1)) * p2 - np.sqrt(j / (j + 1)) * p3

            pp = np.sqrt(2 * n) * p2
            z1 = z
            z = z1 - p1 / pp

            if np.abs(z - z1) <= EPS:
                break

        x[i] = z
        x[n - 1 - i] = -z
        w[i] = 2.0 / pp ** 2
        w[n - 1 - i] = w[i]
    return x, w


def psi(x, t, phi_x):
    """Approximation of wavepacket by GAUSSIAN quadrature procedure"""
    x_N, w = gauss_hermite()
    F = 0.0

    for j in range(0, len(x_N), 1):
        F += w[j] * phi_x(x, t, x_N[j])
    return F


def info(psi):
    """Some information about scattering probabilities for the console."""
    print('\nAnalytical\n'
          f'Reflection probability: {round(np.abs(B / A) ** 2, 4)}\n'
          f'Transmission probability: {round(np.abs(F / A) ** 2, 4)}')
    print('\nGAUSS-QUAD')
    wf.prob_info(psi)
