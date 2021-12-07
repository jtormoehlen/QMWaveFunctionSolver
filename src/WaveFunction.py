import numpy as np
import scipy.constants as const

# Initial values: position x_0, energy E_0, energy uncertainty DeltaE
# barrier width a and height V_0, mass m and momentums p_0, kappa_0.
# a = 1.
# V_0 = 1.
E_0 = 0.9

m_e = const.m_e
V_b = 0.5 * const.eV
a_b = 10. * const.angstrom
E_e = E_0 * V_b
hbar = const.hbar

kb_a = np.sqrt(2. * m_e * V_b) * a_b / hbar
# m = kb_a ** 2

p_ = np.sqrt(2. * m_e * E_e)

# sigma_E = E_e / 100.
# sigma_p = np.sqrt(2. * m_e * sigma_E)
sigma_p = p_ / 20
sigma_x = hbar / sigma_p
x_0 = -a_b - 5. * sigma_x

psi0_norm = 1. / (2. * np.pi * sigma_x ** 2) ** 0.25

# Discretization of spatial coords
x_max = -2. * x_0
n_x = 1000
x, dx = np.linspace(-x_max, x_max, n_x, retstep=True)


def psi_0(x):
    """Initial wave function psi of free particle centered at x_0 with average momentum p_0."""
    psi = np.zeros(x.size, complex)
    for i in range(x.size):
        psi[i] = psi0_norm * np.exp(-(x[i] - x_0) ** 2 / (sigma_x ** 2)) * np.exp(1j * p_ / hbar * x[i])
    return psi


def psi_t(t, p):
    """Time dependent solution of schrodinger equation psi_t."""
    return np.exp(-1j * (p ** 2 / (2. * m_e * hbar)) * t)


def V(x):
    """Time independent potential V(x)."""
    V_N = np.zeros(x.size)
    for index, value in enumerate(x):
        if -a_b <= value <= a_b:
            V_N[index] = V_b
    return V_N


def t_col(p=p_):
    """Collision time from particle with initial position x_0 and momentum p_0."""
    return -x_0 / (p / m_e)


def norm(psi):
    """Normalization of wave function |psi|^2."""
    return np.sum(np.abs(psi) ** 2 * dx)


def prob(psi2, x_start=-x_max, x_end=x_max):
    """Probability of finding the particle in [x_start, x_end] based on formula: sum{|psi|^2*dx}."""
    P = 0.0
    for index, value in enumerate(x):
        if x_start <= value <= x_end:
            P += psi2[index] * dx
    return P


def param_info():
    """Some information about initial parameters for the console."""
    print('Parameters######################\n'
          f'Barrier strength: {round(kb_a, 2)}\n'
          f'E/V-ratio: {round(E_e / V_b, 2)}\n'
          f'Initial position: {round(x_0, 2)}\n'
          '################################')


def prob_info(psi):
    """Prototype for logging scattering probabilities."""
    psi2_norm = norm(psi)
    refl = prob(np.abs(psi) ** 2 / psi2_norm, -x_max, -a_b)
    trans = prob(np.abs(psi) ** 2 / psi2_norm, a_b, x_max)
    print(f'Reflection probability: {round(refl, 4)}\n'
          f'Transmission probability: {round(trans, 4)}')
