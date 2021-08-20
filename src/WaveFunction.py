import numpy as np

# h = 6.62e-34
h = 1.
h_bar = h / (2. * np.pi)


def potential_step(E, x, t=0):
    # m_e = 9.11e-31
    m_e = 1.
    V_0 = E * 2.
    p = np.sqrt(2. * m_e * E)
    p_ = 1j * np.sqrt(2. * m_e * (V_0 - E))
    x_0 = 0.
    A_2 = 2.
    B_1 = 1.
    if x < 0:
        psi_x = np.exp((1j * p * (x - x_0)) / h_bar) + B_1 * np.exp((1j * p * (x - x_0)) / h_bar)
    else:
        psi_x = A_2 * np.exp((1j * p_ * (x - x_0)) / h_bar)
    # psi_t = np.exp((-1j * E * t) / h_bar)
    return psi_x


def potential_wall(E, x, t=0):
    m_e = 1.
    V_0 = 2.
    d = .5
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
