import numpy as np
import scipy.constants as const

from lib.WaveFunction import Model

a_b = 5*const.angstrom  # barrier width
V_b = 0.25*const.eV  # barrier height V_b
m_e = const.m_e  # electron mass
hbar = const.hbar  # planck's reduced constant

eta = np.sqrt(2*m_e*V_b)*a_b/hbar  # barrier strength
m = eta**2  # mass

E = 0.9  # E_0/V_0
p0 = np.sqrt(m*E)  # momentum
sigp = p0/10  # momentum width
sigx = 1/sigp  # position width
x0 = -1-5*sigx  # initial position

# discrete spatial coords (-2*x_0,-2*x_0+dx,...,2*x_0)
xmax = -2*x0
nx = 1000
x, dx = np.linspace(-xmax, xmax, nx, retstep=True)

# Collision time t_col with barrier for classical particle
# :param p: momentum
# :return: collision time
t0 = lambda p: -x0/(p/m)
# discrete time coords (0,dt,2*dt,...,t0)
tn = t0(p0)
dt = tn / 100
t = np.arange(0.0, tn, dt)

K = 1/(np.pi*sigx)**0.25
C = 2*sigp/(2*np.pi*sigp)**0.25

class Barrier(Model):

    def __init__(self, name):
        self.name = name
        self.x, self.dx = x, dx
        self.t, self.dt = t, dt
        self.m, self.E = m, E
        self.sigp = sigp
        self.p0, self.sigp = p0, sigp
        self.C = C

    # Initial wave packet psi(x,0) with average momentum p_0
    # :param x: spatial coords
    # :return: initial wave packet
    def psi0(self, x, p=p0):
        psi = np.zeros(x.size, complex)
        for i in range(x.size):
            psi[i] = K*np.exp(-(x[i]-x0)**2/(sigx**2))*np.exp(1j*p*(x[i]-x0))
        return psi

    # Time dependent solution of schrodinger equation tau(t)
    # :param t: time coord
    # :param p: momentum
    # :return: time dependent solution
    def tau(self, t, p):
        return np.exp(-1j*(p**2/m)*t)

    # Rectangle barriere
    # :return: potential
    def V(self, x):
        return np.heaviside(1-np.abs(x), 1.0)
    
    # Stationary solution (scattering by a rectangle-barrier) superposed with tau
    def phi(self, x, t, p):

        # dimensionless wave numbers k_0 and kappa_0
        if self.E <= 1.0:
            k0 = np.sqrt(self.m*(1-self.E))
        else:
            k0 = 1j*np.sqrt(self.m*(self.E-1))

        # coefficients of stationary solution (scattering on a barrier)
        A = 1.0
        F = A/((np.cosh(2*k0)+(1j/2)*(k0/p0-p0/k0)*np.sinh(2*k0))*np.exp(2j*p0))
        B = F*(-1j/2)*(k0/p0+p0/k0)*np.sinh(2*k0)
        C = (F/2)*(1-(1j*p0/k0))*np.exp(k0+1j*p0)
        D = (F/2)*(1+(1j*p0/k0))*np.exp(-k0+1j*p0)

        psif = np.zeros(x.size, complex)
        if p**2 <= m:
            k = np.sqrt(m-p**2)
        else:
            k = 1j*np.sqrt(p**2-m)
        t = t-t0(p)  # set time t->t-t_col

        for i in range(x.size):
            if x[i] < -1.0:  # reflection region
                psif[i] = (A * np.exp(1j*p*x[i])+B*np.exp(-1j*p*x[i]))
            elif -1.0 <= x[i] <= 1.0:  # barrier region
                psif[i] = (C*np.exp(-k*x[i])+D*np.exp(k*x[i]))
            elif x[i] > 1.0:  # transmission region
                psif[i] = F*np.exp(1j*p*x[i])
        return psif*self.tau(t, p)  # superpose time-dependent solution
    
    def params(self):
        print('NULL')

    # Scattering probabilities
    # :param psi: wave function
    def probs(self, psi):
        norm = Model.prob(np.abs(psi)**2, self.x, self.dx, min(self.x), max(self.x))
        refl = Model.prob(np.abs(psi)**2/norm, self.x, self.dx, -xmax, -1.0)
        trans = Model.prob(np.abs(psi)**2/norm, self.x, self.dx, 1.0, xmax)
        print(f'Reflection probability: {round(refl, 4)}\n'
            f'Transmission probability: {round(trans, 4)}')
        
    # def x_t(self, t):
    #     x_pos = x_0+(2*p0/m)*t  # type: ignore # pos at time t
    #     if x_pos >= -1.0:
    #         t_col = (-1-x0)/(2*p0/m)
    #         return -1-(2*p0/m)*(t-t_col)
    #     else:
    #         return x_pos