import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from lib import WFNumeric as wn, WFAnalytic as wa, WFGeneral as wg

plt.style.use('./lib/figstyle.mpstyle')
FPS = 10  # frames per second


def main(save=False):
    """
    1.) Compute |psi(x,0)|^2 by GH; |psi(x,t)|^2 by RK and CN
    2.) Plot |psi(x,0)|^2 as [Line2D,...]
    3.) Compute norm |psi(x,0)|^2
    :param save: animation if true
    """
    fig = plt.figure()
    ax = plt.subplot()

    x = wg.x_j  # spatial coords
    t = wn.t_n  # time coords

    gh = wa.psi(x, 0)
    psi_gh, = ax.plot(x, np.abs(gh) ** 2, label='Gauss-Hermite')

    wg.params()
    wa.probs(wa.psi(x, t[-1]))

    rk = wn.RKSolver()
    psi_rk, = ax.plot(x, np.abs(rk.psi(0)) ** 2, label='Runge-Kutta')
    cn = wn.CNSolver()
    psi_cn, = ax.plot(x, np.abs(cn.psi(0)) ** 2, label='Crank-Nicolson')

    p, = ax.plot(wg.x_0, 0., 'ok')  # classical p position x(0)

    def init():
        """
        Plot potential V(x).
        :return: wave packets as [[Line2D,...],[Line2D,...],...]
        """
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(-0.1 * max(np.abs(gh) ** 2),
                    max(np.abs(gh) ** 2) + 0.1 * max(np.abs(gh) ** 2))
        ax.set_xlabel('Position $x/$a')
        ax.set_ylabel('Probability density $|\psi(x,t)|^2/$a$^{-1}$')
        ax.plot(x, wg.V(x), '--k')
        return psi_gh, psi_rk, psi_cn, p

    def update(i):
        """
        1.) Compute normalized |psi(x,i)|^2 by GH, RK and CN.
        2.) Plot |psi(x,i)|^2 as [Line2D,...].
        :param i: time (frame) in (t_0,t_1,...,t_N)
        :return: wave packets as [[Line2D,...],[Line2D,...],...]
        """
        psi2_gh = np.abs(wa.psi(x, t[i])) ** 2
        psi_gh.set_ydata(psi2_gh)
        psi2_rk = np.abs(rk.psi(i)) ** 2
        psi_rk.set_ydata(psi2_rk)
        psi2_cn = np.abs(cn.psi(i)) ** 2
        psi_cn.set_ydata(psi2_cn)

        p.set_xdata(wa.x_t(t[i]))
        return psi_gh, psi_rk, psi_cn, p

    plt.legend()
    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(t), interval=100, blit=True)

    if save:
        anim.save('./wave_packet.gif', writer=PillowWriter(fps=FPS),
                  progress_callback=lambda i, n: print(f'Saving frame {i + 1} of {n}'))
        sys.exit(0)
    else:
        plt.show()


if __name__ == "__main__":
    main(save=False)
