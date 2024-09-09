import numpy as np
from lib.WaveFunction import Model

# Computation of GAUSS-HERMITE abscissas and weights
x_H, w = np.polynomial.hermite.hermgauss(200)
K = 2*Model.sigma_p/(2*np.pi*Model.sigma_p)**0.25

class GHSolver(Model):

    def __init__(self, Model):
        self.E = Model.E
        self.m = Model.m
        # dimensionless wave numbers k_0 and kappa_0
        p0 = Model.p0
        if self.E <= 1.0:
            k0 = np.sqrt(self.m*(1-self.E))
        else:
            k0 = 1j*np.sqrt(self.m*(self.E-1))

        # coefficients of stationary solution (scattering on a barrier)
        self.A = 1.0
        self.F = self.A/((np.cosh(2*k0)+(1j/2)*(k0/p0-p0/k0)*np.sinh(2*k0))*np.exp(2j*p0))
        self.B = self.F*(-1j/2)*(k0/p0+p0/k0)*np.sinh(2*k0)
        self.C = (self.F/2)*(1-(1j*p0/k0))*np.exp(k0+1j*p0)
        self.D = (self.F/2)*(1+(1j*p0/k0))*np.exp(-k0+1j*p0)

        self.p0 = p0
        self.k0 = k0

    # Stationary solution (scattering by a rectangle-barrier) superposed with psi_t
    def phi_alpha(self, x, t, p=Model.p0):
        psi_xt = np.zeros(x.size, complex)
        if p**2 <= Model.m:
            k = np.sqrt(Model.m-p**2)
        else:
            k = 1j*np.sqrt(p**2-Model.m)
        t = t-Model.t_0(p)  # set time t->t-t_col

        for i in range(x.size):
            if x[i] < -1.0:  # reflection region
                psi_xt[i] = (self.A * np.exp(1j*p*x[i])+self.B*np.exp(-1j*p*x[i]))
            elif -1.0 <= x[i] <= 1.0:  # barrier region
                psi_xt[i] = (self.C*np.exp(-k*x[i])+self.D*np.exp(k*x[i]))
            elif x[i] > 1.0:  # transmission region
                psi_xt[i] = self.F*np.exp(1j*p*x[i])
        return psi_xt*Model.psi_t(t, p)  # superpose time-dependent solution

    # Approximation of wavepacket by GAUSS-HERMITE procedure
    def psi(self, x, t, phi=phi_alpha):
        psi = 0
        for j in range(0, len(x_H), 1):
            p = 2*Model.sigma_p*x_H[j]+Model.p0  # momentum substitution for GH-proc
            psi += K*w[j]*phi(x, t, p)
        return psi

    # Scattering probabilities
    def probs(self, psi):
        print('\nAnalytical\n'
            f'Reflection probability: {round(np.abs(self.B/self.A)**2, 4)}\n'
            f'Transmission probability: {round(np.abs(self.F/self.A)**2, 4)}')
        print('\nGAUSS-HERMITE')
        Model.probs(psi)

    # Position of classical particle x(t)
    # :param t: time
    # :return: position
    def x_t(self, t):
        x_0 = Model.x0  # start pos
        x_pos = x_0+(2*p0/m)*t  # type: ignore # pos at time t
        if x_pos >= -1.0:
            t_col = (-1-x_0)/(2*Model.p0/Model.m)
            return -1-(2*Model.p0/Model.m)*(t-t_col)
        else:
            return x_pos

    def v_x(self, x):
        E = Model.p0** 2/Model.m
        V = Model.V(x)
        return 0.0