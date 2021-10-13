import WFAnimation as wfa
import WFPotential as wfp

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.sparse import linalg as ln
from scipy import sparse as sparse
from scipy import fft, ifft


class WavePacket:
    def __init__(self):
        n_points = 1000
        self.x_begin = -200.0
        self.x_end = 200.0
        self.sigma0 = 5.0
        self.E0 = 10.0
        self.k0 = np.sqrt(2 * self.E0)
        self.x0 = -150.0
        self.dt = 0.5
        self.prob = sp.zeros(n_points)

        """ 1) Space discretization """
        self.x, self.dx = sp.linspace(self.x_begin, self.x_end, n_points, retstep=True)

        """ 2) Initialization of the wave function to Gaussian wave packet """
        norm = (2.0 * sp.pi * self.sigma0 ** 2) ** (-0.25)
        self.psi = sp.exp(-(self.x - self.x0) ** 2 / (4.0 * self.sigma0 ** 2))
        self.psi = self.psi * sp.exp(1.0j * self.k0 * self.x)
        self.psi = self.psi * (2.0 * sp.pi * self.sigma0 ** 2) ** (-0.25)

        """ 3) Setting up the potential barrier """
        self.potential = wfp.wall(self.x)

        """ 4) Creating the Hamiltonian """
        h_diag = sp.ones(n_points) / self.dx ** 2 + self.potential
        h_non_diag = sp.ones(n_points - 1) * (-0.5 / self.dx ** 2)
        hamiltonian = sparse.diags([h_diag, h_non_diag, h_non_diag], [0, 1, -1])

        """ 5) Computing the Crank-Nicolson time evolution matrix """
        implicit = (sparse.eye(n_points) - self.dt / 2.0j * hamiltonian).tocsc()
        explicit = (sparse.eye(n_points) + self.dt / 2.0j * hamiltonian).tocsc()
        self.evolution_matrix = ln.inv(implicit).dot(explicit).tocsr()

    def evolve(self):
        self.psi = self.evolution_matrix.dot(self.psi)
        self.prob = abs(self.psi) ** 2
        self.norm = sum(self.prob)
        self.prob /= self.norm
        self.psi /= self.norm ** 0.5
        # tmp_v = 0.0
        # tmp_i = 0.0
        # for index, value in enumerate(self.prob):
        #     if value >= tmp_v:
        #         tmp_v = value
        #         tmp_i = index
        # self.x_pos = x[tmp_i]
        return np.real(self.psi)


wave_packet = WavePacket()
animator = wfa.Animator(wave_packet)
animator.animate()
plt.show()
