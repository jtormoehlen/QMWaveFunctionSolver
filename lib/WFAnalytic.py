import numpy as np

# Computation of GAUSS-HERMITE abscissas and weights
hx, w = np.polynomial.hermite.hermgauss(200)

class GHSolver():

    def __init__(self, model):
        self.model = model

    # Approximation of wavepacket by GAUSS-HERMITE procedure
    def psi(self, t):
        psi = 0
        for j in range(0, len(hx), 1):
            p = 2*self.model.sigp*hx[j]+self.model.p0  # momentum substitution for GH-proc
            psi += self.model.C*w[j]*self.model.phi(self.model.x, t, p)
        return psi

    # Scattering probabilities
    def probs(self, psi):
        # print('\nAnalytical\n'
        #     f'Reflection probability: {round(np.abs(self.B/self.A)**2, 4)}\n'
        #     f'Transmission probability: {round(np.abs(self.F/self.A)**2, 4)}')
        print('\nGAUSS-HERMITE')
        self.model.probs(psi)

    # Position of classical particle x(t)
    # :param t: time
    # :return: position
    def x_t(self, t):
        pass