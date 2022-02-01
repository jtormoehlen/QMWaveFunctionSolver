import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
from scipy import integrate
from scipy.sparse import linalg as ln
import scipy.constants as const


class Psi:
    def __init__(self, E_0):
        self.a_b = 5. * const.angstrom
        self.V_b = 0.25 * const.eV

        self.kb_a = np.sqrt(2. * const.m_e * self.V_b) * self.a_b / const.hbar
        self.m = self.kb_a ** 2

        self.E_0 = E_0
        self.k0 = np.sqrt(self.m * self.E_0)
        if self.E_0 <= 1.:
            self.kappa0 = np.sqrt(self.m * (1. - self.E_0))
        else:
            self.kappa0 = 1j * np.sqrt(self.m * (self.E_0 - 1.))

        self.sigma_p = self.k0 / 15
        self.sigma_x = 1 / self.sigma_p
        self.x_0 = -1 - 5 * self.sigma_x

        self.x_max = -2 * self.x_0
        self.n_x = 1000
        self.x, self.dx = np.linspace(-self.x_max, self.x_max, self.n_x, retstep=True)

        self.t_0 = 0.
        self.t_n = self.t_col()
        self.dt = self.t_n / 100
        self.t_N = np.arange(self.t_0, self.t_n, self.dt)

    def cn(self):
        H = self.hamilton()
        implicit = (sp.sparse.eye(self.x.size) - self.dt / 2j * H).tocsc()
        explicit = (sp.sparse.eye(self.x.size) + self.dt / 2j * H).tocsc()
        evolution_matrix = ln.inv(implicit).dot(explicit).tocsr()
        psi0 = self.psi_0(self.x)
        psi_norm = self.norm(psi0)
        delta_t = 0.0
        while delta_t <= self.t_n:
            psi0 = evolution_matrix.dot(psi0)
            delta_t += self.dt
        return self.prob(np.abs(psi0) ** 2 / psi_norm, 1., self.x_max)

    def rk(self):
        rk = integrate.solve_ivp(self.eq, y0=self.psi_0(self.x),
                                 t_span=[self.t_0, self.t_n], t_eval=self.t_N, method='RK23')
        psi = np.abs(rk.y[:, self.t_N.size - 1]) ** 2
        psi_norm = self.norm(rk.y[:, 0])
        return self.prob(psi / psi_norm, -self.x_max, -1.)

    def prob(self, psi2, x_start, x_end):
        P = 0.0
        for index, value in enumerate(self.x):
            if x_start <= value <= x_end:
                P += psi2[index] * self.dx
        return P

    def norm(self, psi):
        return np.sum(np.abs(psi) ** 2 * self.dx)

    def eq(self, t, psi):
        H = self.hamilton()
        return -1j * H.dot(psi)

    def psi_0(self, x):
        psi = np.zeros(x.size, complex)
        for i in range(x.size):
            psi[i] = np.exp(-(x[i] - self.x_0) ** 2 / (self.sigma_x ** 2)) * np.exp(1j * self.k0 * x[i])
        return psi

    def V(self, x):
        V_N = np.zeros(x.size)
        for index, value in enumerate(x):
            if -1. <= value <= 1.:
                V_N[index] = 1.
        return V_N

    def hamilton(self):
        H_diag = sp.ones(self.x.size) * 2. / (self.m * self.dx ** 2) + self.V(self.x)
        H_non_diag = sp.ones(self.x.size - 1) * -1. / (self.m * self.dx ** 2)
        return sp.sparse.diags([H_non_diag, H_diag, H_non_diag], [-1, 0, 1])

    def t_col(self):
        return -self.x_0 / (self.k0 / self.m)

    def trans(self):
        eta = (self.kappa0 / self.k0) - (self.k0 / self.kappa0)
        # return (1 + (1 + 0.25 * eta ** 2) * np.sin(2 * self.kappa0 * self.a_b) ** 2) ** (-1.)
        A = 1.
        F = A / ((np.cosh(2 * self.kappa0) + (1j / 2) * eta * np.sinh(2 * self.kappa0)) * np.exp(2j * self.k0))
        return np.abs(F / A) ** 2


fig = plt.figure(figsize=(6, 4))
ax = plt.subplot(1, 1, 1)
E_num = np.linspace(0.01, 8.01, 50)
E_ana = np.linspace(0.01, 8.01, 100)
R_cn = np.zeros(E_num.size)
R_rk = np.zeros(E_num.size)
R_ana = np.zeros(E_ana.size)

for index, value in enumerate(E_ana):
    psi = Psi(value)
    R_ana[index] = psi.trans()

for index, value in enumerate(E_num):
    psi = Psi(value)
    R_cn[index] = psi.cn()
    # R_rk[index] = psi.rk()

ax.set_xlabel(r'Energy ratio $E_0/V_0$')
# ax.set_ylabel(r'Reflection probability $R$')
# ax.set_xlabel(r'Momentum deviation $\sigma_p$')
ax.set_ylabel(r'Transmission probability $T$')
ax.plot(E_ana, R_ana, label='analytical')
ax.plot(E_num, R_cn, 'p', label='Crank-Nicolson')
# ax.plot(E_num, R_rk, '*', label='Runge-Kutta')
# ax.set_ylim(0., 1.)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
