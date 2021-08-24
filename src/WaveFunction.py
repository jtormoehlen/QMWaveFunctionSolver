import numpy as np
import WFTool as wft

# h = 6.62e-34
h = 1.
h_bar = h / (2. * np.pi)
# m_e = 9.11e-31
m_e = 1.


class Potential:
    @staticmethod
    def inf_pot(E, a, x):
        if 0 <= x <= a:
            A_1 = np.sqrt(2. / a)
            A_2 = -A_1
            n = np.sqrt((2. * m_e * a ** 2 * E)) / (np.pi * h_bar)
            k = (np.floor(n) * np.pi) / a
            psi_x = A_1 * np.exp(1j * k * x) + A_2 * np.exp(-1j * k * x)
        else:
            psi_x = 0
        return psi_x + E

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
        psi_t = np.exp((-1j * E * t) / h_bar)
        return psi_x * psi_t + E

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
        psi_t = np.exp((-1j * E * t) / h_bar)
        return psi_x * psi_t + E

    @staticmethod
    def fin_pot(E, V_0, d, x, t=0):
        if V_0 <= E:
            k_1 = k_3 = np.sqrt((2. * m_e * (E - V_0)) / h_bar ** 2)
            k_2 = np.sqrt((2. * m_e * E) / h_bar ** 2)
            A_1_1 = 1.
            A_1_2 = ((A_1_1 * (k_1 - k_2) * (k_1 + k_2) * np.sin(d * k_2)) / (
                    (k_1 ** 2 + k_2 ** 2) * np.sin(d * k_2) + 2j * k_1 * k_2 * np.cos(d * k_2)))
            A_2_1 = ((-2 * A_1_1 * k_1 * (k_1 + k_2)) / (-(k_1 + k_2) ** 2 + np.exp(2j * d * k_2) * (k_1 - k_2) ** 2))
            A_2_2 = ((2 * A_1_1 * k_1 * np.exp(2j * d * k_2) * (k_1 - k_2)) / (
                    -(k_1 + k_2) ** 2 + np.exp(2j * d * k_2) * (k_1 - k_2) ** 2))
            A_3_1 = ((-4 * A_1_1 * k_1 * k_2 * np.exp(-1j * d * (k_1 - k_2))) / (
                    -(k_1 + k_2) ** 2 + np.exp(2j * d * k_2) * (k_1 - k_2) ** 2))
        else:
            k_1 = k_3 = 1j * np.sqrt((2. * m_e * (V_0 - E)) / h_bar ** 2)
            k_2 = np.sqrt((2. * m_e * E) / h_bar ** 2)
            A_1_1 = A_3_2 = 0.
            A_2_1 = 1.
            A_1_2 = (2. * k_2 * A_2_1) / (k_2 - k_1)
            A_2_2 = ((k_2 + k_1) * A_2_1) / (k_2 - k_1)
            A_3_1 = (2. * k_2) / (k_2 + k_1)
        if x < 0:
            psi_x = A_1_1 * np.exp(1j * k_1 * x) + A_1_2 * np.exp(-1j * k_1 * x)
        elif 0 <= x <= d:
            psi_x = A_2_1 * np.exp(1j * k_2 * x) + A_2_2 * np.exp(-1j * k_2 * x)
        else:
            psi_x = A_3_1 * np.exp(1j * k_3 * x)
        psi_t = np.exp((-1j * E * t) / h_bar)
        return psi_x * psi_t

    @staticmethod
    def harmonic(omega, x, t=0):
        const = ((m_e * omega) / (np.pi * h_bar)) ** (1. / 4.)
        psi_x = const * np.exp((-m_e * omega * x ** 2) / (2. * h_bar))
        V = (m_e * omega ** 2 * x ** 2) / 2.
        return psi_x, V

    @staticmethod
    def wave_package(x, t=0):
        A = 2.
        delta_p = .5
        p_x = 1.e1
        psi_x = A * (2. / (np.pi * delta_p ** 2)) ** (1. / 4.) * np.exp(-x ** 2 / delta_p ** 2) * np.exp(
            1j * x * (p_x / h_bar) - (2 * np.pi * t))
        return psi_x
