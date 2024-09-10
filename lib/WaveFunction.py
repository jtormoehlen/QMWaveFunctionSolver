import numpy as np

class Model:

    def __init__(self, name):
        self.name = name
        self.x, self.t = np.linspace(0.0, 1.0, 100), np.linspace(0.0, 1.0, 100)

    # Initial wave packet psi(x,0) with average momentum p_0
    # :param x: spatial coords
    # :return: initial wave packet
    def psi0(self, x, p):
        pass

    # Time dependent solution of schrodinger equation psi(t)
    # :param t: time coord
    # :param p: momentum
    # :return: time dependent solution
    def tau(self, t, p):
        pass

    # Time independent potential V(x)
    # :param x: spatial coord
    # :return: potential
    def V(self, x):
        pass

    def phi(self, x, t, p):
        pass
    
    # Scattering probabilities
    def probs(self, psi):
        pass

    # Probability of finding the particle in selected interval based on formula: sum{|psi|^2*dx}
    # :param psi2: normalized probability density
    # :param x_start: lower position boundary
    # :param x_end: upper position boundary
    # :return: probability sum{|psi|^2*dx} from a to b
    @staticmethod
    def prob(psi2, x, dx, a, b):
        P = 0.0
        for index, value in enumerate(x):
            if a <= value <= b:
                P += psi2[index]*dx
        return P