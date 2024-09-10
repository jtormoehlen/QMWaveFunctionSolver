import scipy as sp
from scipy import integrate
from scipy.sparse import linalg as ln

# Discrete HAMILTON-operator
def hop(x, dx, m, V):
    d0 = sp.ones(x.size)*2/(m*dx**2)+V(x)
    d1 = sp.ones(x.size-1)*-1/(m*dx**2)
    return sp.sparse.diags([d1, d0, d1], [-1, 0, 1])

class RKSolver():

    # Solution of schrodinger-equation by RUNGE-KUTTA procedure for initial free wave packet
    def __init__(self, model):
        self.model = model

        # Right-hand side of schrodinger-equation
        def seq(t, psi):
            return -1j*(hop(model.x, model.dx, model.m, model.V)).dot(psi)

        self.sol = integrate.solve_ivp(seq, y0=model.psi0(model.x), 
                                       t_span=[min(model.t), max(model.t)], 
                                       t_eval=model.t, method='RK23')
        self.probs()

    # Solution psi(x,t)
    # :param t: time index in (0,1,...,|t_n|)
    # :return: wave function
    def psi(self, t):
        return self.sol.y[:, t]

    # Scattering probabilities
    def probs(self):
        print('\nRUNGE-KUTTA')
        self.model.probs(self.psi(-1))


class CNSolver():

    # Solution of schrodinger-equation by CRANK-NICOLSON procedure for initial free wave packet
    def __init__(self, model):
        self.model = model

        # Evolve psi^n_j to psi^n_j+1
        def evolve(sol):
            return self.evolMat.dot(sol)

        H = hop(model.x, model.dx, model.m, model.V)
        impMat = (sp.sparse.eye(model.x.size)-model.dt/2j*H).tocsc()
        expMat = (sp.sparse.eye(model.x.size)+model.dt/2j*H).tocsc()
        self.evolMat = ln.inv(impMat).dot(expMat).tocsr()
        self.sol = [model.psi0(model.x)]
        for t in range(model.t.size):
            self.sol.append(evolve(self.sol[-1]))
        self.probs()

    # Solution psi(x,t)
    # :param t: time index in (0,1,...,|t_n|)
    # :return: wave function
    def psi(self, t):
        return self.sol[t]

    # Scattering probabilities
    def probs(self):
        print('\nCRANK-NICOLSON')
        self.model.probs(self.psi(-1))
