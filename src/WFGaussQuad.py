import numpy as np


def gauss_hermite():
    n = 100
    EPS = 1.0e-14
    x, w = np.zeros(n), np.zeros(n)
    m = int((n + 1) / 2)
    z = pp = 1.0

    for i in range(0, m, 1):
        if i == 0:
            z = np.sqrt((2 * n + 1) - 1.85575 * (2 * n + 1) ** -0.16667)
        elif i == 1:
            z -= 1.14 * (n ** 0.426) / z
        elif i == 2:
            z = 1.86 * z - 0.86 * x[0]
        elif i == 3:
            z = 1.91 * z - 0.91 * x[1]
        else:
            z = 2.0 * z - x[i - 2]

        for its in range(0, n, 1):
            p1 = 1 / np.pi ** 0.25
            p2 = 0.0

            for j in range(0, n, 1):
                p3 = p2
                p2 = p1
                p1 = z * np.sqrt(2.0 / (j + 1)) * p2 - np.sqrt(j / (j + 1)) * p3

            pp = np.sqrt(2 * n) * p2
            z1 = z
            z = z1 - p1 / pp

            if np.abs(z - z1) <= EPS:
                break

        x[i] = z
        x[n - 1 - i] = -z
        w[i] = 2.0 / pp ** 2
        w[n - 1 - i] = w[i]
    return x, w


def gauss_quad(x, t, f):
    x_N, w = gauss_hermite()
    F = 0.0

    for j in range(0, len(x_N), 1):
        F = F + w[j] * f(x, t, x_N[j])
    return F
