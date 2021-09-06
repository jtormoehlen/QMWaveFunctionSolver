import numpy as np

# h = 6.62e-34
h = 1.
h_bar = h / (2. * np.pi)
# m_e = 9.11e-31
m_e = 1.


def spectral(p, p_0, sigma_p):
    c = 1. / ((2. * np.pi) ** (1./4.) * np.sqrt(sigma_p))
    return c * np.exp(-(p - p_0) ** 2 / (4. * sigma_p ** 2))


def step(E, V_0, x, t=0):
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
    return psi_x * psi_t + E


def psi_t(E, t):
    return np.exp((-1j * E * t) / h_bar)


class Potential:

    @staticmethod
    def step(E, V_0, x, t=0):
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
        return psi_x * psi_t(E, t)

    @staticmethod
    def wall(E, V_0, d, x, t=0):
        k_1 = k_3 = np.sqrt((2. * m_e * E) / h_bar ** 2)
        if V_0 <= E:
            k_2 = np.sqrt((2. * m_e * (E - V_0)) / h_bar ** 2)
        else:
            k_2 = 1j * np.sqrt((2. * m_e * (V_0 - E)) / h_bar ** 2)
        A_1_1 = 1.
        A_1_2 = ((A_1_1 * (k_1 - k_2) * (k_1 + k_2) * np.sin(d * k_2)) / (
                    (k_1 ** 2 + k_2 ** 2) * np.sin(d * k_2) + 2j * k_1 * k_2 * np.cos(d * k_2)))
        A_2_1 = ((-2 * A_1_1 * k_1 * (k_1 + k_2)) / (-(k_1 + k_2) ** 2 + np.exp(2j * d * k_2) * (k_1 - k_2) ** 2))
        A_2_2 = ((2 * A_1_1 * k_1 * np.exp(2j * d * k_2) * (k_1 - k_2)) / (
                    -(k_1 + k_2) ** 2 + np.exp(2j * d * k_2) * (k_1 - k_2) ** 2))
        A_3_1 = ((-4 * A_1_1 * k_1 * k_2 * np.exp(-1j * d * (k_1 - k_2))) / (
                    -(k_1 + k_2) ** 2 + np.exp(2j * d * k_2) * (k_1 - k_2) ** 2))
        if x < 0:
            psi_x = A_1_1 * np.exp(1j * k_1 * x) + A_1_2 * np.exp(-1j * k_1 * x)
        elif 0 <= x <= d:
            psi_x = A_2_1 * np.exp(1j * k_2 * x) + A_2_2 * np.exp(-1j * k_2 * x)
        else:
            psi_x = A_3_1 * np.exp(1j * k_3 * x)
        return psi_x * psi_t(E, t)

    @staticmethod
    def wave_packet(p_0, V_0, N, x, t=0):
        # A = 2.
        # delta_p = .5
        # p_x = 1.e1
        # psi_x = A * (2. / (np.pi * delta_p ** 2)) ** (1. / 4.) * np.exp(-x ** 2 / delta_p ** 2) * np.exp(
        #     1j * x * (p_x / h_bar) - (2 * np.pi * t))
        # return psi_x
        # c0 = 2.
        # lambda0 = 1.
        # y1 = -(x - (c0 * t)) ** 2
        # y2 = 2 * np.pi * ((x - (c0 * t)) / lambda0)
        # u = np.exp(y1) * np.exp(1j * y2)
        E_0 = p_0 ** 2 / (2. * m_e)
        delta_p = 0.5
        sigma_p = 0.1
        psi_n = 0.
        x_0 = 0.
        for n in np.linspace(-N, N, 2 * N + 1):
            p_n = p_0 + n * delta_p
            f = spectral(p_n, p_0, sigma_p)
            psi_n += f * step(E_0, V_0, x - x_0, t) * delta_p
        return psi_n

