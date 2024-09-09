import numpy as np
import scipy.constants as const

a_b = 5*const.angstrom  # barrier width
V_b = 0.25*const.eV  # barrier height V_b
m_e = const.m_e  # electron mass
hbar = const.hbar  # planck's reduced constant

kb_a = np.sqrt(2*m_e*V_b)*a_b/hbar  # barrier strength
m = kb_a**2  # mass

E = 0.9  # E_0/V_0
p0 = np.sqrt(m*E)  # momentum
sigma_p = p0/10  # momentum width
sigma_x = 1/sigma_p  # position width
x0 = -1-5*sigma_x  # initial position

# discrete spatial coords (-2*x_0,-2*x_0+dx,...,2*x_0)
x_max = -2*x0
n_x = 1000
x_j, dx = np.linspace(-x_max, x_max, n_x, retstep=True)

# Collision time t_col with barrier for classical particle
# :param p: momentum
# :return: collision time
def t_0(p=p0):
    return -x0 / (p / m)

# # discrete time coords (0,dt,2*dt,...,t_0)
t_N = t_0()
dt = t_N / 100
t_n = np.arange(0.0, t_N, dt)

K = 1/(np.pi*sigma_x)**0.25

# Initial wave packet psi(x,0) with average momentum p_0
# :param x: spatial coords
# :return: initial wave packet
def psi_0(x):
    psi = np.zeros(x.size, complex)
    for i in range(x.size):
        psi[i] = K*np.exp(-(x[i]-x0)**2/(sigma_x**2))*np.exp(1j*p0*(x[i]-x0))
    return psi

# Time dependent solution of schrodinger equation psi(t)
# :param t: time coord
# :param p: momentum
# :return: time dependent solution
def psi_t(t, p):
    return np.exp(-1j*(p**2/m)*t)

# Time independent potential V(x)
# :param x: spatial coord
# :return: potential
def V(x):
    pass

# Probability of finding the particle in selected interval based on formula: sum{|psi|^2*dx}
# :param psi2: normalized probability density
# :param x_start: lower position boundary
# :param x_end: upper position boundary
# :return: probability sum{|psi|^2*dx} from a to b
def prob(psi2, x_start=-x_max, x_end=x_max):
    P = 0.
    for index, value in enumerate(x_j):
        if x_start <= value <= x_end:
            P += psi2[index]*dx
    return P

# Information about initial parameters for the console
def params():
    pass

# Scattering probabilities
# :param psi: wave function
def probs(psi):
    psi2_norm = prob(np.abs(psi)**2)
    refl = prob(np.abs(psi)**2/psi2_norm, -x_max, -1.0)
    trans = prob(np.abs(psi)**2/psi2_norm, 1.0, x_max)
    print(f'Reflection probability: {round(refl, 4)}\n'
          f'Transmission probability: {round(trans, 4)}')
    
class Model:
    a_b = 5*const.angstrom  # barrier width
    V_b = 0.25*const.eV  # barrier height V_b
    m_e = const.m_e  # electron mass
    hbar = const.hbar  # planck's reduced constant

    kb_a = np.sqrt(2*m_e*V_b)*a_b/hbar  # barrier strength
    m = kb_a**2  # mass

    E = 0.9  # E_0/V_0
    p0 = np.sqrt(m*E)  # momentum
    sigma_p = p0/10  # momentum width
    sigma_x = 1/sigma_p  # position width
    x0 = -1-5*sigma_x  # initial position

    # discrete spatial coords (-2*x_0,-2*x_0+dx,...,2*x_0)
    x_max = -2*x0
    n_x = 1000
    x, dx = np.linspace(-x_max, x_max, n_x, retstep=True)

    K = 1/(np.pi*sigma_x)**0.25

    # Collision time t_col with barrier for classical particle
    # :param p: momentum
    # :return: collision time
    def t_0(p=p0):
        return -x0 / (p / m)

    # # discrete time coords (0,dt,2*dt,...,t_0)
    t_N = t_0()
    dt = t_N / 100
    t_n = np.arange(0.0, t_N, dt)

    def __init__(self, name):
        self.name = name

    # Initial wave packet psi(x,0) with average momentum p_0
    # :param x: spatial coords
    # :return: initial wave packet
    def psi_0(x):
        psi = np.zeros(x.size, complex)
        for i in range(x.size):
            psi[i] = K*np.exp(-(x[i]-x0)**2/(sigma_x**2))*np.exp(1j*p0*(x[i]-x0))
        return psi

    # Time dependent solution of schrodinger equation psi(t)
    # :param t: time coord
    # :param p: momentum
    # :return: time dependent solution
    def psi_t(t, p):
        return np.exp(-1j*(p**2/m)*t)

    # Time independent potential V(x)
    # :param x: spatial coord
    # :return: potential
    def V(x):
        return np.heaviside(1. - np.abs(x), 1.)

    # Probability of finding the particle in selected interval based on formula: sum{|psi|^2*dx}
    # :param psi2: normalized probability density
    # :param x_start: lower position boundary
    # :param x_end: upper position boundary
    # :return: probability sum{|psi|^2*dx} from a to b
    def prob(psi2, x_start=-x_max, x_end=x_max):
        P = 0.
        for index, value in enumerate(x_j):
            if x_start <= value <= x_end:
                P += psi2[index]*dx
        return P

    # Information about initial parameters for the console
    def params(self):
        pass

    # Scattering probabilities
    # :param psi: wave function
    def probs(psi):
        psi2_norm = prob(np.abs(psi)**2)
        refl = prob(np.abs(psi)**2/psi2_norm, -x_max, -1.0)
        trans = prob(np.abs(psi)**2/psi2_norm, 1.0, x_max)
        print(f'Reflection probability: {round(refl, 4)}\n'
            f'Transmission probability: {round(trans, 4)}')