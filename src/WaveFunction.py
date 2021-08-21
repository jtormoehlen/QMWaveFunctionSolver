import numpy as np

# h = 6.62e-34
h = 1.
h_bar = h / (2. * np.pi)


def potential_step(E, V_0, x, t=0):
    # m_e = 9.11e-31
    m_e = 1.
    k = np.sqrt((2. * m_e * E) / h_bar ** 2)
    if V_0 <= E:
        q = np.sqrt((2. * m_e * (E - V_0)) / h_bar ** 2)
        R = (k - q) / (k + q)
        T = (2. * k) / (k + q)
    else:
        kappa = np.sqrt((2. * m_e * (V_0 - E)) / h_bar ** 2)
        q = -kappa / 1j
        R = (k - (1j * kappa)) / (k + (1j * kappa))
        T = (2. * k) / (k + (1j * kappa))
    if x < 0:
        psi_x = np.exp(1j * k * x) + R * np.exp(-1j * k * x)
    else:
        psi_x = T * np.exp(1j * q * x)
    psi_t = np.exp((-1j * E * t) / h_bar)
    return psi_x * psi_t


def potential_wall(E, V_0, d, x, t=0):
    m_e = 1.
    p = np.sqrt(2. * m_e * E)
    p_ = 1j * np.sqrt(2. * m_e * (V_0 - E))
    x_0 = .0
    A_2 = 1.
    A_3 = .5
    B_1 = 1.
    B_2 = 1.
    if x < 0:
        psi_x = np.exp((1j * p * (x - x_0)) / h_bar) + B_1 * np.exp((1j * p * (x - x_0)) / h_bar)
    elif 0 <= x <= d:
        psi_x = A_2 * np.exp((1j * p_ * (x - x_0)) / h_bar) + B_2 * np.exp((1j * p_ * (x - x_0)) / h_bar)
    elif x > d:
        psi_x = A_3 * np.exp((1j * p * (x - x_0)) / h_bar)
    else:
        psi_x = 2.
    return psi_x
