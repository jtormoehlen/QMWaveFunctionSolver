import numpy as np
import scipy.constants as const

# Barrier width a_b, barrier height V_b, electron mass m_e and planck's constant hbar
a_b = 5. * const.angstrom
V_b = 0.25 * const.eV
m_e = const.m_e
hbar = const.hbar

# barrier strength and dimensionless mass number
kb_a = np.sqrt(2. * m_e * V_b) * a_b / hbar
m = kb_a ** 2

# Initial values: energy E -> E_0/V_0, momentum p_0, sigma_p -> 10% of p_0
E = 0.9
p_0 = np.sqrt(m * E)
sigma_p = p_0 / 10
sigma_x = 1. / sigma_p
x_0 = -1 - 5 * sigma_x

# Discretization of spatial coords from -2*x_0 to 2*x_0
x_max = -2. * x_0
n_x = 2000
x, dx = np.linspace(-x_max, x_max, n_x, retstep=True)


def psi_0(x):
    """
    Initial wave function psi of free particle with average momentum p_0.
    :param x: particle centered at x_0
    :return: initial wave packet
    """
    psi = np.zeros(x.size, complex)
    for i in range(x.size):
        psi[i] = np.exp(-(x[i] - x_0) ** 2 / (sigma_x ** 2)) * np.exp(1j * p_0 * x[i])
    return psi


def psi_t(t, p):
    """
    Time dependent solution of schrodinger equation psi_t.
    :param t: time variable
    :param p: momentum
    :return: time dependent solution
    """
    return np.exp(-1j * (p ** 2 / m) * t)


def V(x):
    """
    Time independent potential V(x).
    :param x: position variable
    :return: potential
    """
    return np.heaviside(1. - np.abs(x), 1.)


def t_col(p=p_0):
    """
    Collision time t_col from particle with initial position x_0.
    :param p: momentum
    :return: collision time
    """
    return -x_0 / (p / m)


def norm(psi):
    """
    Norm of wave function |psi|^2.
    :param psi: wave function
    :return: norm
    """
    return np.sum(np.abs(psi) ** 2 * dx)


def prob(psi2, x_start=-x_max, x_end=x_max):
    """
    Probability of finding the particle in selected interval based on formula: sum{|psi|^2*dx}.
    :param psi2: normalized probability density
    :param x_start: lower position boundary
    :param x_end: upper position boundary
    :return: probability sum{|psi|^2*dx} from a to b
    """
    P = 0.0
    for index, value in enumerate(x):
        if x_start <= value <= x_end:
            P += psi2[index] * dx
    return P


def param_info():
    """
    Some information about initial parameters for the console.
    """
    print('Parameters######################\n'
          f'Barrier strength: {round(kb_a, 2)}\n'
          f'E/V-ratio: {round(E, 2)}\n'
          f'Initial position: {round(x_0, 2)}\n'
          '################################')


def prob_info(psi):
    """
    Prototype for logging scattering probabilities.
    :param psi: wave function
    """
    psi2_norm = norm(psi)
    refl = prob(np.abs(psi) ** 2 / psi2_norm, -x_max, -1.)
    trans = prob(np.abs(psi) ** 2 / psi2_norm, 1., x_max)
    print(f'Reflection probability: {round(refl, 4)}\n'
          f'Transmission probability: {round(trans, 4)}')
