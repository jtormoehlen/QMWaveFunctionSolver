import numpy as np
from scipy.integrate import quad

# h = 6.62e-34
h = 1.
h_bar = h / (2. * np.pi)
# m_e = 9.11e-31
m = 1.


def M(x_0, v_0, sigma_x, x, t):
    c = 1. / (((2. * np.pi) ** (1. / 4.)) * np.sqrt(sigma_x))
    if v_0 < 0:
        return heaviside(-x) * c * np.exp(-(x - x_0 - (v_0 * t)) ** 2 / (4. * sigma_x ** 2))
    else:
        return c * np.exp(-(x - x_0 - (v_0 * t)) ** 2 / (4. * sigma_x ** 2))


def phi(x_0, p_0, v_0, sigma_x, sigma_p, x, t):
    alpha = np.arctan((2. * sigma_p ** 2 * t) / (h_bar * m))
    return (k_eff(x_0, p_0, v_0, sigma_x, sigma_p, x, t) * (x - x_0 - (v_0 * t))) + ((p_0 * v_0 * t) / (2. * h_bar)) - (alpha / 2.)


def k_eff(x_0, p_0, v_0, sigma_x, sigma_p, x, t):
    k = (1. / h_bar) * (p_0 + ((sigma_p ** 2 * t) / (2. * m * sigma_x ** 2)) * (x - x_0 - (v_0 * t)))
    return k


def heaviside(x):
    if x >= 0:
        return 1.
    else:
        return 0.


def psi_1_plus(E, x):
    epsilon = (2. * m * E) / h_bar ** 2
    k_1 = np.sqrt(epsilon)
    A_1 = np.sqrt(m / (2. * np.pi * h_bar ** 2 * k_1))
    return heaviside(-x) * A_1 * np.exp(1j * k_1 * x)


def psi_1_minus(E, V_0, x):
    U_0 = (2. * m * V_0) / h_bar ** 2
    epsilon = (2. * m * E) / h_bar ** 2
    k_1 = np.sqrt(epsilon)
    A_1 = np.sqrt(m / (2. * np.pi * h_bar ** 2 * k_1))
    if E >= V_0:
        k_2 = np.sqrt(epsilon - U_0)
        B_1 = ((k_1 - k_2) * A_1) / (k_1 + k_2)
    else:
        kappa_2 = np.sqrt(U_0 - epsilon)
        B_1 = ((k_1 - 1j * kappa_2) * A_1) / (k_1 + 1j * kappa_2)
    return heaviside(-x) * B_1 * np.exp(-1j * k_1 * x)


def psi_2(E, V_0, x):
    U_0 = (2. * m * V_0) / h_bar ** 2
    epsilon = (2. * m * E) / h_bar ** 2
    k_1 = np.sqrt(epsilon)
    A_1 = np.sqrt(m / (2. * np.pi * h_bar ** 2 * k_1))
    if E >= V_0:
        k_2 = np.sqrt(epsilon - U_0)
        A_2 = (2. * k_1 * A_1) / (k_1 + k_2)
        psi = heaviside(x) * A_2 * np.exp(1j * k_2 * x)
    else:
        kappa_2 = np.sqrt(U_0 - epsilon)
        A_2 = (2. * k_1 * A_1) / (1 + 1j * kappa_2)
        psi = heaviside(x) * A_2 * np.exp(-kappa_2 * x)
    return psi


def psi_t(E, t):
    return np.exp((-1j * E * t) / h_bar)


class Potential:
    @staticmethod
    def step(E, V_0, x, t=0):
        k = np.sqrt((2. * m * E) / h_bar ** 2)
        if V_0 <= E:
            q = np.sqrt((2. * m * (E - V_0)) / h_bar ** 2)
            R = (k - q) / (k + q)
            T = (2. * k) / (k + q)
        else:
            kappa = np.sqrt((2. * m * (V_0 - E)) / h_bar ** 2)
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
        k_1 = k_3 = np.sqrt((2. * m * E) / h_bar ** 2)
        if V_0 <= E:
            k_2 = np.sqrt((2. * m * (E - V_0)) / h_bar ** 2)
        else:
            k_2 = 1j * np.sqrt((2. * m * (V_0 - E)) / h_bar ** 2)
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
    def wave_packet(E_0, V_0, x, t=0):
        sigma_p = 0.2
        sigma_x0 = h_bar / (2. * sigma_p)
        sigma_v = sigma_p / 1.
        sigma_x = np.sqrt(sigma_x0 ** 2 + (sigma_v * t) ** 2)
        x_0 = -3.
        p_0 = np.sqrt(2. * m * E_0)
        v = p_0 / m
        # for n in np.linspace(-N, N, (2 * N) + 1):
        #     p_n = p_0 + (n * delta_p)
        #     E_n = p_n ** 2 / (2. * m_e)
        #     psi_n += M(x_0, p_0 / 1., sigma_x, x, t) * step(E_n, V_0, x - x_0, t) * delta_p
        M_func = M(x_0, v, sigma_x, x, t)
        M_func_1_minus = M(-x_0, -v, sigma_x, x, t)
        phi_func = np.exp(1j * phi(x_0, p_0, v, sigma_x, sigma_p, x, t))
        psi_1_plus_func = psi_1_plus(E_0, x)
        psi_1_minus_func = psi_1_minus(E_0, V_0, x)
        psi_2_func = psi_2(E_0, V_0, x)
        psi_n = (M_func * psi_1_plus_func + M_func_1_minus * psi_1_minus_func + M_func * psi_2_func) * psi_t(E_0, t)
        return psi_n
