import numpy as np

class Model:

    def __init__(self, name):
        self.name = name

    # initial wave packet psi(x,0) with average momentum p0
    # x: spatial coords
    # return: initial wave packet
    def psi0(self, x, p):
        pass

    # time-dependent solution of schrodinger equation psi(t)
    # t: time coord
    # p: momentum
    # return: time dependent solution
    def tau(self, t, p):
        pass

    # time-independent potential V(x)
    # param x: spatial coords
    # return: potential
    def V(self, x):
        pass

    # stationary solution
    # x: spatial coords
    # t: time coord
    # p: momentum
    def phi(self, x, t, p):
        pass
    
    # scattering probabilities
    def probs(self, psi):
        pass

    # probability of finding given particle in selected interval based on sum{|psi|^2*dx}
    # psi2: normalized probability density
    # a: lower boundary
    # b: upper boundary
    # return: summed up probability
    @staticmethod
    def prob(psi2, x, dx, a, b):
        P = 0.0
        for index, value in enumerate(x):
            if a <= value <= b:
                P += psi2[index]*dx
        return P
    
    @staticmethod
    def normalize(psi, x, dx, a, b):
        norm = Model.prob(np.abs(psi)**2, x, dx, a, b)
        return psi/np.sqrt(norm)