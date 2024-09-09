import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from lib import WFNumeric as wn, WFAnalytic as wa
from lib.WaveFunction import Model

plt.style.use('./lib/figstyle.mpstyle')
FPS = 10  # frames per second

# Start animation
# :param save: animation if true
def main(save):
    # Compute |psi(x,0)|^2 by GH; |psi(x,t)|^2 by RK and CN
    # Plot |psi(x,0)|^2 as [Line2D,...]
    # Compute norm |psi(x,0)|^2
    fig = plt.figure()
    ax = plt.subplot()

    model = Model("Barriere")

    x = model.x  # spatial coords
    t = model.t_n  # time coords

    gh = wa.GHSolver(model)
    psi_gh, = ax.plot(x, np.abs(gh.psi(x, 0, gh.phi_alpha)) ** 2, label='Gauss-Hermite')

    model.params()
    gh.probs(gh.psi(x, t[-1], gh.phi_alpha))

    rk = wn.RKSolver(model)
    psi_rk, = ax.plot(x, np.abs(rk.psi(0)) ** 2, label='Runge-Kutta')
    cn = wn.CNSolver(model)
    psi_cn, = ax.plot(x, np.abs(cn.psi(0)) ** 2, label='Crank-Nicolson')

    p, = ax.plot(model.x0, 0., 'ok')  # classical p position x(0)

    # Plot potential V(x)
    # :return: wave packets as [[Line2D,...],[Line2D,...],...]
    def init():
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(-0.1 * max(np.abs(gh.psi(x, 0, gh.phi_alpha)) ** 2),
                    max(np.abs(gh.psi(x, 0, gh.phi_alpha)) ** 2) + 0.1 * max(np.abs(gh.psi(x, 0, gh.phi_alpha)) ** 2))
        ax.set_xlabel('Position $x/$a')
        ax.set_ylabel('Probability density $|\psi(x,t)|^2/$a$^{-1}$')
        ax.plot(x, Model.V(x), '--k')
        return psi_gh, psi_rk, psi_cn, p

    # 1.) Compute normalized |psi(x,i)|^2 by GH, RK and CN
    # 2.) Plot |psi(x,i)|^2 as [Line2D,...]
    # :param i: time (frame) in (t_0,t_1,...,t_N)
    # :return: wave packets as [[Line2D,...],[Line2D,...],...]
    def update(i):
        psi2_gh = np.abs(gh.psi(x, t[i], gh.phi_alpha)) ** 2
        psi_gh.set_ydata(psi2_gh)
        psi2_rk = np.abs(rk.psi(i)) ** 2
        psi_rk.set_ydata(psi2_rk)
        psi2_cn = np.abs(cn.psi(i)) ** 2
        psi_cn.set_ydata(psi2_cn)

        # p.set_xdata(gh.x_t(t[i]))
        return psi_gh, psi_rk, psi_cn,

    plt.legend()
    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(t), interval=100, blit=True)

    if save:
        anim.save('./img/'+model.name+'.gif', writer=PillowWriter(fps=FPS),
                  progress_callback=lambda i, n: print(f'Saving frame {i + 1} of {n}'))
        sys.exit(0)
    else:
        plt.show()

if __name__ == '__main__':
    main(save=True)