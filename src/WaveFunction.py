import numpy as np

# Discretization of spatial coords
x_max = 100
n_x = 1000
x, dx = np.linspace(-x_max, x_max, n_x, retstep=True)

# Initial values: position x_0, energy E_0, energy uncertainty DeltaE
# barrier width a and height V_0, mass m and momentums p_0, kappa_0.
a = 1.0
x_0 = -50
V_0 = 1.0
E_0 = 0.3
m = 0.5

sigma_E = E_0 / 100
sigma_p = np.sqrt(m * sigma_E)

p_ = np.sqrt(m * E_0)
kappa_ = np.sqrt(m * (1.0 - E_0))


def psi_0(x, t, p=0):
    """Initial wave function psi of free particle centered at x_0 with average momentum p_0."""
    psi = np.zeros(x.size, complex)
    p_0 = 2 * sigma_p * p + p_
    for i in range(x.size):
        psi[i] = np.exp(1j * p_0 * (x[i] - x_0))
    return psi * psi_t(t, p_0)


def psi_t(t, p):
    """Time dependent solution of schrodinger equation psi_t."""
    return np.exp(-1j * (p ** 2 / m) * t)


def V(x):
    """Time independent potential V(x)."""
    V_N = np.zeros(x.size)
    for index, value in enumerate(x):
        if -a <= value <= a:
            V_N[index] = V_0
    return V_N


def t_col(p=p_):
    """Collision time from particle with initial position x_0 and momentum p_0."""
    return (a - x_0) / (p / m)


def norm(psi):
    """Normalization of wave function |psi|^2."""
    return np.sum(np.abs(psi) ** 2 * dx)


def prob(psi2, x_start, x_end):
    """Probability of finding the particle in [x_start, x_end] based on formula: sum{|psi|^2*dx}."""
    P = 0.0
    for index, value in enumerate(x):
        if x_start <= value <= x_end:
            P += psi2[index] * dx
    return P


def param_info():
    """Some information about initial parameters for the console."""
    print('Parameters######################\n'
          f'Mass number: {m}\n'
          f'Energy level: {E_0 / V_0} V_0\n'
          f'Initial position: {x_0} a\n'
          f'Barrier width: {2 * a} a\n'
          '################################')


def prob_info(psi):
    psi2_norm = norm(psi)
    refl = prob(np.abs(psi) ** 2 / psi2_norm, -x_max, -a)
    trans = prob(np.abs(psi) ** 2 / psi2_norm, a, x_max)
    print(f'Reflection probability: {round(refl, 4)}\n'
          f'Transmission probability: {round(trans, 4)}')
