import numpy as np
import scipy.constants as const

# Initial values: position x_0, energy E_0, energy uncertainty DeltaE
# barrier width a and height V_0, mass m and momentums p_0, kappa_0.
a_b = 5. * const.angstrom
V_b = 0.25 * const.eV
m_e = const.m_e
hbar = const.hbar
kb_a = np.sqrt(2. * m_e * V_b) * a_b / hbar
m = kb_a ** 2

E = 0.9
p_0 = np.sqrt(m * E)
sigma_p = p_0 / 10
sigma_x = 1. / sigma_p
x_0 = -1 - 5 * sigma_x

# Discretization of spatial coords
x_max = -2. * x_0
n_x = 1000
x, dx = np.linspace(-x_max, x_max, n_x, retstep=True)


def psi_0(x):
    """Initial wave function psi of free particle centered at x_0 with average momentum p_0."""
    psi = np.zeros(x.size, complex)
    for i in range(x.size):
        psi[i] = np.exp(-(x[i] - x_0) ** 2 / (sigma_x ** 2)) * np.exp(1j * p_0 * x[i])
    return psi


def psi_t(t, p):
    """Time dependent solution of schrodinger equation psi_t."""
    return np.exp(-1j * (p ** 2 / m) * t)


def V(x):
    """Time independent potential V(x)."""
    V = np.zeros(x.size)
    for index, value in enumerate(x):
        if -1. <= value <= 1.:
            V[index] = 1.
    return V


def t_col(p=p_0):
    """Collision time from particle with initial position x_0 and momentum p_0."""
    return -x_0 / (p / m)


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
          f'E/V-ratio: {round(E, 2)}\n'
          f'Initial position: {round(x_0, 2)}\n'
          '################################')


def prob_info(psi):
    """Prototype for logging scattering probabilities."""
    psi2_norm = norm(psi)
    refl = prob(np.abs(psi) ** 2 / psi2_norm, -x_max, -1.)
    trans = prob(np.abs(psi) ** 2 / psi2_norm, 1., x_max)
    print(f'Reflection probability: {round(refl, 4)}\n'
          f'Transmission probability: {round(trans, 4)}')
