import numpy as np
from scipy.integrate import quad

# h = 6.62e-34
h = 1.
h_bar = h / (2. * np.pi)
# m_e = 9.11e-31
m = 1.


def M(x_0, v_0, sigma_x, x, t, E_0=0, V_0=0):
    c = 1. / (((2. * np.pi) ** (1. / 4.)) * np.sqrt(sigma_x))
    if v_0 < 0:
        return theta(-x) * c * np.exp(-(x - x_0 - (v_0 * t)) ** 2 / (4. * sigma_x ** 2))
    else:
        # x_l = x_0 - sigma_x
        # if x_l + (v_0 * t) + (2. * sigma_x) >= 0:
        #     if E_0 > V_0:
        #         v_0 = np.sqrt(2. * ((E_0 - V_0) / m))
        return c * np.exp(-(x - x_0 - (v_0 * t)) ** 2 / (4. * sigma_x ** 2))


def theta(x):
    if x >= 0:
        return 1.
    else:
        return 0.


def free(E, x):
    k = np.sqrt((2. * m * E) / h_bar ** 2)
    return np.exp(1j * k * x)


def step_free(E, x):
    return theta(-x) * free(E, x)


def step_reflect(E, V_0, x):
    k = np.sqrt((2. * m * E) / h_bar ** 2)
    if E > V_0:
        q = np.sqrt((2. * m * (E - V_0)) / h_bar ** 2)
        R = (k - q) / (k + q)
    else:
        kappa = np.sqrt((2. * m * (V_0 - E)) / h_bar ** 2)
        R = (k - (1j * kappa)) / (k + (1j * kappa))
    return theta(-x) * R * np.exp(-1j * k * x)


def step_trans(E, V_0, x):
    k = np.sqrt((2. * m * E) / h_bar ** 2)
    if E > V_0:
        q = np.sqrt((2. * m * (E - V_0)) / h_bar ** 2)
        T = (2. * k) / (k + q)
        return theta(x) * T * np.exp(1j * q * x)
    else:
        kappa = np.sqrt((2. * m * (V_0 - E)) / h_bar ** 2)
        T = (2. * k) / (k + (1j * kappa))
        return theta(x) * T * np.exp(-kappa * x)


def wall_free(E, a, x):
    if x < -a:
        return free(E, x)
    else:
        return 0.


def wall_reflect(E, V_0, a, x):
    if x < -a:
        k = np.sqrt((2. * m * E) / h_bar ** 2)
        kappa = np.sqrt((2. * m * (V_0 - E)) / h_bar ** 2)
        eta = (kappa / k) + (k / kappa)
        F = wall_F(E, V_0, a)
        B = F * ((-1j * eta) / 2.) * np.sinh(2. * kappa * a)
        return theta(-x) * B * np.exp(-1j * k * x)
    else:
        return 0.


def wall_barr(E, V_0, a, x):
    if -a <= x <= a:
        k = np.sqrt((2. * m * E) / h_bar ** 2)
        kappa = np.sqrt((2. * m * (V_0 - E)) / h_bar ** 2)
        F = wall_F(E, V_0, a)
        C = (1. / 2.) * (1 - ((1j * kappa) / kappa)) * np.exp((kappa + 1j * k) * a) * F
        D = (1. / 2.) * (1 + ((1j * kappa) / kappa)) * np.exp((-kappa + 1j * k) * a) * F
        return C * np.exp(-kappa * x) + D * np.exp(kappa * x)
    else:
        return 0.


def wall_trans(E, V_0, a, x):
    if x > a:
        k = np.sqrt((2. * m * E) / h_bar ** 2)
        F = wall_F(E, V_0, a)
        return F * np.exp(1j * k * x)
    else:
        return 0.


def wall_F(E, V_0, a):
    A = 1.0
    k = np.sqrt((2. * m * E) / h_bar ** 2)
    kappa = np.sqrt((2. * m * (V_0 - E)) / h_bar ** 2)
    epsilon = (kappa / k) - (k / kappa)
    return A / ((np.cosh(2. * kappa * a) + ((1j * epsilon) / 2.) * np.sinh(2. * kappa * a)) * np.exp(2j * k * a))


def psi_t(E, t):
    return np.exp((-1j * E * t) / h_bar)


class Potential:
    @staticmethod
    def classical(E, V_0, t=0):
        x_0 = -5.
        v_0 = np.sqrt((2. * E) / m)
        x = (x_0 + (v_0 * t))
        if x < 0:
            return x
        else:
            delta_t = -x_0 / v_0
            if E >= V_0:
                v = np.sqrt((2. * (E - V_0)) / m)
                return v * (t - delta_t)
            else:
                return -v_0 * (t - delta_t)

    @staticmethod
    def step(E, V_0, x, t=0):
        psi_x = step_free(E, x) + step_reflect(E, V_0, x) + step_trans(E, V_0, x)
        return psi_x * psi_t(E, t)

    @staticmethod
    def wall(E, V_0, a, x, t=0):
        psi_x = wall_free(E, a, x) + wall_reflect(E, V_0, a, x) + wall_barr(E, V_0, a, x) + wall_trans(E, V_0, a, x)
        return psi_x * psi_t(E, t)

    @staticmethod
    def step_wave_packet(E_0, V_0, x, t=0):
        sigma_x0 = 0.5
        sigma_p = h_bar / (2. * sigma_x0)
        sigma_v = sigma_p / 1.
        sigma_x = np.sqrt(sigma_x0 ** 2 + (sigma_v * t) ** 2)
        x_0 = -5.
        p_0 = np.sqrt(2. * m * E_0)
        v_0 = p_0 / m
        M_free = M(x_0, v_0, sigma_x, x, t, E_0)
        M_reflect = M(-x_0, -v_0, sigma_x, x, t, E_0)
        M_trans = M(x_0, v_0, sigma_x, x, t, E_0, V_0)
        psi_n = (M_free * step_free(E_0, x) + M_reflect * step_reflect(E_0, V_0, x) + M_trans * step_trans(E_0, V_0, x)) * psi_t(E_0, t)
        # psi_n = (M_free * step_free(E_0, x) + M_trans * step_trans(E_0, V_0, x)) * psi_t(E_0, t)
        return psi_n

    @staticmethod
    def wall_wave_packet(E_0, V_0, a, x, t=0):
        sigma_x0 = 0.5
        sigma_p = h_bar / (2. * sigma_x0)
        sigma_v = sigma_p / m
        sigma_x = np.sqrt(sigma_x0 ** 2 + (sigma_v * t) ** 2)
        x_0 = -5.
        p_0 = np.sqrt(2. * m * E_0)
        v_0 = p_0 / m
        M_free = M(x_0, v_0, sigma_x, x, t)
        M_reflect = M(-x_0, -v_0, sigma_x, x, t)
        psi_n = (M_free * wall_free(E_0, a, x) + M_reflect * wall_reflect(E_0, V_0, a, x) + M_free * wall_barr(E_0, V_0, a, x) + M_free * wall_trans(E_0, V_0, a, x)) * psi_t(E_0, t)
        return psi_n
