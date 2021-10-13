import numpy as np
import WaveFunctionTest as wft

m = h_bar = 1
t = 0
x = np.arange(-10, 10, 0.01)
x_0 = -5
E_0 = 60
V_0 = 50
p_0 = np.sqrt(2 * m * E_0)

if E_0 >= V_0:
    p = np.sqrt(p_0 ** 2 - 2 * m * V_0)
else:
    p = 1j * np.sqrt(2 * m * V_0 - p_0 ** 2)

sigma_E = E_0 / 100
sigma_p = np.sqrt(2 * m * sigma_E)
sigma_x0 = h_bar / (2 * sigma_p)
sigma_v = sigma_p / m


def sigma_x(t):
    return np.sqrt(sigma_x0 ** 2 + (sigma_v * t) ** 2)


def f_x(x, t, x0, p):
    A = 1 / ((2 * np.pi) ** 0.25 * np.sqrt(sigma_x(t)))
    return A * np.exp(-(x - x0 - (p / m) * t) ** 2 / (4 * sigma_x(t) ** 2))


def psi_trans(x, t):
    result = np.zeros(x.size, complex)
    T = (2 * p) / (p + p_0)
    for i in range(x.size):
        if x[i] < 0:
            result[i] = np.exp((1j / h_bar) * p_0 * (x[i] - x_0)) * psi_t(t, p_0) * f_x(x[i], t, x_0, p_0)
        else:
            result[i] = T * np.exp((1j / h_bar) * p * (x[i] - p / p_0 * x_0)) * psi_t(t, p) * f_x(x[i], t, p / p_0 * x_0, p)
            # result[i] = T * np.exp((1j / h_bar) * p_ * (x[i])) * psi_t(t, p_) * f_x(x[i], t, (p_ / p_0) * x_0, p_)
            # if x_0 + p_0 * t < 0:
            #     result[i] = T * np.exp((1j / h_bar) * p * (x[i] - x_0)) * psi_t(t, p) * f_x(x[i], t, x_0, p_0)
            # else:
            #     result[i] = T * np.exp((1j / h_bar) * p * (x[i] - p / p_0 * x_0)) * psi_t(t, p) * f_x(x[i], t, p / p_0 * x_0, p)
    return result


def psi_reflect(x, t):
    result = np.zeros(x.size, complex)
    R = (p - p_0) / (p + p_0)
    for i in range(x.size):
        if x[i] < 0:
            result[i] = R * np.exp((-1j / h_bar) * p_0 * (x[i] + x_0)) * psi_t(t, p_0) * f_x(x[i], t, -x_0, -p_0)
    return result


def psi_t(t, p):
    return np.exp((-1j * (p ** 2 / (2 * m)) * t) / h_bar)


def f_p(p):
    A = 1 / ((2 * np.pi) ** 0.25 * np.sqrt(sigma_p))
    return A * np.exp(-(p - p_0) ** 2 / (4 * sigma_p ** 2))


def psi_free(x, t):
    result = np.zeros(x.size, complex)
    for i in range(x.size):
        result[i] = np.exp((1j / h_bar) * p_0 * (x[i] - x_0)) * psi_t(t, p_0) * f_x(x[i], t, x_0, p_0)
    return result


def psi_N(x, t):
    N = 15
    delta_p = 0.1
    psi_N = np.zeros(x.size, complex)
    # for n in np.linspace(-N, N, int(2 * N + 1)):
    #     p_n = p_0 + n * delta_p
    #     p_n_ = p_x(1) + n * delta_p
    #     psi_N = psi_N + f_p(p_n) * (psi_1a(x, t, p_n) + psi_1b(x, t, p_n) + psi_2(x, t, p_n_)) * delta_p
    psi_N = psi_trans(x, t) + psi_reflect(x, t)
    # psi_N = psi_free(x, t)
    return psi_N


def V(x):
    V_N = np.zeros(x.size)
    for i, v in enumerate(x):
        if v >= 0:
            V_N[i] = V_0
    return V_N


def x_p(t):
    x_pos = x_0 + (p_0 / m) * t
    if x_pos < 0:
        return x_pos
    else:
        dx = (p / p_0) * x_0
        return dx + (p / m) * t


def prob():
    return ((2 * p) / (p + p_0)) ** 2


wft.animate(x)
