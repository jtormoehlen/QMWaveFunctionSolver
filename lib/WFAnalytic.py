import numpy as np

# import Gauss-Hermite abscissas and weights from NumPy
hx, w = np.polynomial.hermite.hermgauss(200)

class GHSolver():

    # wave packet approximated by Gauss-Hermite integration of energy eigenfunctions
    def __init__(self, model):
        self.model = model

    # wave packet approximated by Gauss-Hermite integration
    # t: time coord
    # return: wave packet
    def psi(self, t):
        psi = 0
        for j in range(0, len(hx), 1):
            p = 2*self.model.sigp*hx[j]+self.model.p0  #
            psi += self.model.C*w[j]*self.model.phi(self.model.x, t, p)
        return psi

    # scattering probabilities
    def probs(self, psi):
        # print('\nAnalytical\n'
        #     f'Reflection probability: {round(np.abs(self.B/self.A)**2, 4)}\n'
        #     f'Transmission probability: {round(np.abs(self.F/self.A)**2, 4)}')
        print('\nGAUSS-HERMITE')
        self.model.probs(psi)

    # classical particle position x(t)
    # t: time coord
    # return: position
    def x_t(self, t):
        pass