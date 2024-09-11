import numpy as np

from lib.WaveFunction import Model

m = 1.0 # mass [hbar^2/(2a^2 E_1)]
p0 = 4*np.pi**3 # avg init momentum [hbar/a]
x0 = 0.5 # avg init position [a]
sigp = p0/10 # momentum width
sigx = 1/sigp # position width
V0 = 1.0E6 # potential height

# discretized spatial coords (-2*x0,-2*x0+dx,...,2*x0)
a = 0.0
b = 1.0
x1 = a-0.1
xn = b+0.1
nx = 1000
x, dx = np.linspace(x1, xn, nx, retstep=True)

t0 = lambda p: x0*m/p
# discretized time coords (0,dt,2*dt,...,t0)
# tn = 4*(1.0-x0)*m/p0
tn = 4*t0(p0)*(1/x0-1)
nt = 500
t, dt = np.linspace(0.0, tn, nt, retstep=True)

# integration constants (auto-normalization)
C = 1.0

class Pot(Model):

    def __init__(self, name):
        self.name = name
        self.x, self.dx, self.x0 = x, dx, x0
        self.t, self.dt = t, dt
        self.m = m
        self.p0, self.sigp = p0, sigp
        self.C = C
        self.xp = x0
        self.xr = False

    # initial wave packet psi(x,0) with average momentum p0
    # x: spatial coords
    # return: initial wave packet
    def psi0(self, x, p=p0):
        psif = np.ones(x.size, complex)*\
            np.exp(-(x-x0)**2/(sigx**2))*np.exp(1j*p*(x-x0))
        return Model.normalize(psif, x, dx, a, b)

    # time-dependent solution of schrodinger equation tau(t)
    # t: time coord
    # p: momentum
    # return: time-dependent solution
    def tau(self, t, p):
        t = t+t0(p)
        return np.exp(-1j*(p**2/m)*t)

    # emulate infinite potential pot with V0>>E
    # return: potential
    def V(self, x):
        V = np.zeros(x.size)
        for i, xi in enumerate(x):
            if xi < 0.0 or xi > 1.0:
                V[i] = V0
        return V

    # stationary solution (particle in pot) superposed with tau(t)
    def phi(self, x, t, p):

        phif = np.zeros(x.size, complex)
        for i in range(x.size):
            if 0.0 < x[i] < 1.0:
                phif[i] = np.sin(p*x[i])
        return phif*self.tau(t, p)
    
    # initial conditions information
    def params(self, psi=None):
        print('Parameters######################\n'
            f'Pot width: {round(b-a, 2)}\n'
            f'Energy ratio E/V: {round(np.sqrt(m*p0)/V0, 2)}\n'
            f'Initial position x0: {round(x0, 2)}\n'
            '################################')
        if psi is not None: self.probs(psi) 

    # scattering probabilities
    # param psi: (un-)normalized wave function
    def probs(self, psi):
        norm = Model.prob(np.abs(psi)**2, x, dx, min(x), max(x))
        print(norm)

    def xpos(self, i):

        def ex(xi):
            for i in range(len(x)):
                if x[i]-dx/2 <= xi <= x[i]+dx/2:
                    e = 2*p0**2/m-self.V(x)[i]
                    if e < 0.0: 
                        self.xr = not self.xr
                    return e

        delta = np.sqrt(2*np.abs(ex(self.xp))/m)*dt
        if not self.xr: 
            self.xp += delta
        else:
            self.xp -= delta
        return self.xp