import numpy as np
import scipy.constants as const

from lib.WaveFunction import Model

a_b = 5*const.angstrom # barrier width
V_b = 0.25*const.eV # barrier height V_b
m_e = const.m_e # electron mass
hbar = const.hbar # reduced plancks constant

eta = np.sqrt(2*m_e*V_b)*a_b/hbar # barrier strength
m = eta**2 # mass

E = 0.9 # E_0/V_0
p0 = np.sqrt(m*E) # momentum
sigp = p0/10 # momentum width
sigx = 1/sigp # position width
x0 = -1-5*sigx # initial position

# discretized spatial coords (-2*x0,-2*x0+dx,...,2*x0)
xn = -2*x0
nx = 1000
x, dx = np.linspace(-xn, xn, nx, retstep=True)

# classical barrier collision time
t0 = lambda p: -x0/(p/m)
# discretized time coords (0,dt,2*dt,...,t0)
tn = t0(p0)
nt = 200
t, dt = np.linspace(0.0, tn, nt, retstep=True)

# integration constants
K = 1/(np.pi*sigx)**0.25
C = 2*sigp/(2*np.pi*sigp)**0.25

class Barrier(Model):

    def __init__(self, name):
        self.name = name
        self.x, self.dx, self.x0 = x, dx, x0
        self.t, self.dt = t, dt
        self.m = m
        self.p0, self.sigp = p0, sigp
        self.C = C

    # initial wave packet psi(x,0) with average momentum p0
    # x: spatial coords
    # return: initial wave packet
    def psi0(self, x, p=p0):
        psif = np.ones(x.size, complex)*\
            np.exp(-(x-x0)**2/(sigx**2))*np.exp(1j*p*(x-x0))
        return Model.normalize(psif, x, dx, -xn, xn)

    # time-dependent solution of schrodinger equation tau(t)
    # t: time coord
    # p: momentum
    # return: time-dependent solution
    def tau(self, t, p):
        return np.exp(-1j*(p**2/m)*t)

    # rectangle barriere using NumPys Heaviside function
    # return: potential
    def V(self, x):
        return np.heaviside(1-np.abs(x), 1.0)
    
    # stationary solution (scattering by rectangle-barrier) superposed with tau(t)
    def phi(self, x, t, p):

        # dimensionless wave number
        k0 = np.sqrt(m*(1-E)) if E <= 1.0 else 1j*np.sqrt(m*(E-1))
        # stationary solution coefficients
        A = 1.0
        F = A/((np.cosh(2*k0)+(1j/2)*(k0/p0-p0/k0)*np.sinh(2*k0))*np.exp(2j*p0))
        B = F*(-1j/2)*(k0/p0+p0/k0)*np.sinh(2*k0)
        C = (F/2)*(1-(1j*p0/k0))*np.exp(k0+1j*p0)
        D = (F/2)*(1+(1j*p0/k0))*np.exp(-k0+1j*p0)

        psif = np.zeros(x.size, complex)
        for i in range(x.size):
            if x[i] < -1.0: # reflection region
                psif[i] = (A * np.exp(1j*p*x[i])+B*np.exp(-1j*p*x[i]))
            elif -1.0 <= x[i] <= 1.0: # barrier region
                k = np.sqrt(m-p**2) if p**2 <= m else 1j*np.sqrt(p**2-m)
                psif[i] = (C*np.exp(-k*x[i])+D*np.exp(k*x[i]))
            elif x[i] > 1.0: # transmission region
                psif[i] = F*np.exp(1j*p*x[i])
        t = t-t0(p) # set time t'->t-t0
        return psif*self.tau(t, p)
    
    # initial conditions information
    def params(self, psi=None):
        print('Parameters######################\n'
            f'Barrier strength eta: {round(eta, 2)}\n'
            f'Ratio E/V: {round(E, 2)}\n'
            f'Initial position x0: {round(x0, 2)}\n'
            '################################')
        if psi is not None: self.probs(psi) 
    

    # scattering probabilities
    # param psi: (un-)normalized wave function
    def probs(self, psi):
        norm = Model.prob(np.abs(psi)**2, x, dx, min(x), max(x))
        refl = Model.prob(np.abs(psi)**2/norm, x, dx, -xn, -1.0)
        trans = Model.prob(np.abs(psi)**2/norm, x, dx, 1.0, xn)
        print(f'Reflection probability: {round(refl, 4)}\n'
            f'Transmission probability: {round(trans, 4)}')
        
    def xpos(self, i):
        pos = x0+(2*p0/m)*t[i]
        if pos >= -1.0:
            tc = (-1-x0)/(2*p0/m)
            return -1-(2*p0/m)*(t[i]-tc)
        else:
            return pos