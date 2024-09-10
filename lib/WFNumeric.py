import scipy as sp
from scipy import integrate
from scipy.sparse import linalg as ln

# discretized Hamiltonian operator
def op(x, dx, m, V):
    d0 = sp.ones(x.size)*2/(m*dx**2)+V(x)
    d1 = sp.ones(x.size-1)*-1/(m*dx**2)
    return sp.sparse.diags([d1, d0, d1], [-1, 0, 1])

class RKSolver():

    # solve schrodinger equation by Runge-Kutta procedure for initial free wave packet
    def __init__(self, model):
        self.model = model

        # right-hand side of schrodinger-equation
        def seq(t, psi):
            return -1j*(op(model.x, model.dx, model.m, model.V)).dot(psi)

        self.sol = integrate.solve_ivp(seq, y0=model.psi0(model.x), 
                                       t_span=[min(model.t), max(model.t)], 
                                       t_eval=model.t, method='RK23')
        self.probs()

    # get solution psi(x,t)
    # i: time index in (0,1,...,|tn|)
    # return: wave function
    def psi(self, i):
        return self.sol.y[:, i]

    # print scattering probabilities
    def probs(self):
        print('\nRUNGE-KUTTA')
        self.model.probs(self.psi(-1))


class CNSolver():

    # solve schrodinger equation by Crank-Nicolson procedure for initial free wave packet
    def __init__(self, model):
        self.model = model

        # evolve psi^n_j to psi^n_j+1
        def evolve(sol):
            return self.evolMat.dot(sol)

        H = op(model.x, model.dx, model.m, model.V)
        impMat = (sp.sparse.eye(model.x.size)-model.dt/2j*H).tocsc()
        expMat = (sp.sparse.eye(model.x.size)+model.dt/2j*H).tocsc()
        self.evolMat = ln.inv(impMat).dot(expMat).tocsr()
        self.sol = [model.psi0(model.x)]
        for t in range(model.t.size):
            self.sol.append(evolve(self.sol[-1]))
        self.probs()

    # get solution psi(x,t)
    # t: time index in (0,1,...,|tn|)
    # return: wave function
    def psi(self, i):
        return self.sol[i]

    # scattering probabilities
    def probs(self):
        print('\nCRANK-NICOLSON')
        self.model.probs(self.psi(-1))
