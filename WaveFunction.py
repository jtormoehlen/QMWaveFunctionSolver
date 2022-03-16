import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from lib import WFNumeric as wn, WFAnalytic as wa, WFGeneral as wg

plt.style.use('./lib/figstyle.mpstyle')
FPS = 10


def main(save=False):
    fig = plt.figure()
    ax = plt.subplot()

    x = wg.x_j  # spatial coords
    t = wn.t_n  # time coords

    """
    1.) compute |psi(x,0)|^2 by GH; |psi(x,t)|^2 by RK and CN
    2.) plot |psi(x,0)|^2 as [Line2D,...]
    3.) compute norm |psi(x,0)|^2
    """
    gh = wa.psi(x, 0)
    psi_gh, = ax.plot(x, np.abs(gh) ** 2, label='Gauss-Hermite')
    norm_gh = wg.prob(np.abs(gh) ** 2)

    wg.params()
    wa.probs(wa.psi(x, t[-1]))

    rk = wn.RKSolver()
    psi_rk, = ax.plot(x, np.abs(rk.psi(0)) ** 2, label='Runge-Kutta')
    norm_rk = wg.prob(np.abs(rk.psi(0)) ** 2)
    cn = wn.CNSolver()
    psi_cn, = ax.plot(x, np.abs(cn.psi(0)) ** 2, label='Crank-Nicolson')
    norm_cn = wg.prob(np.abs(cn.psi(0)) ** 2)

    # prob_text = ax.text(wu.x_0, 0.12, '', fontsize=11)
    p, = ax.plot(wg.x_0, 0., 'ok')  # classical p position x(0)

    ax.set_xlim(min(x), max(x))
    ax.set_ylim(-0.1 * max(np.abs(gh) ** 2 / norm_gh),
                max(np.abs(gh) ** 2 / norm_gh) + 0.1 * max(np.abs(gh) ** 2 / norm_gh))
    ax.set_xlabel('Position $x/$a')
    ax.set_ylabel('Probability density $|\psi(x,t)|^2/$a$^{-1}$')

    def __init():
        """
        plot potential V(x)
        :return: wave packets as [[Line2D,...],[Line2D,...],...]
        """
        ax.plot(x, wg.V(x), '--k')
        return psi_gh, psi_rk, psi_cn, p

    def __update(i):
        """
        1.) compute normalized |psi(x,i)|^2 by GH, RK and CN
        2.) plot |psi(x,i)|^2 as [Line2D,...]
        :param i: time (frame) in (t_0,t_1,...,t_N)
        :return: wave packets as [[Line2D,...],[Line2D,...],...]
        """
        psi2_gh = np.abs(wa.psi(x, t[i])) ** 2 / norm_gh
        psi_gh.set_ydata(psi2_gh)
        psi2_rk = np.abs(rk.psi(i)) ** 2 / norm_rk
        psi_rk.set_ydata(psi2_rk)
        psi2_cn = np.abs(cn.psi(i)) ** 2 / norm_cn
        psi_cn.set_ydata(psi2_cn)

        # prob_text.set_text(r'$P_{GH}=$' + f'{round(wu.prob(psi2_gh), 4)}\n'
        #                    r'$P_{RK}=$' + f'{round(wu.prob(psi2_rk), 4)}\n'
        #                    r'$P_{CN}=$' + f'{round(wu.prob(psi2_cn), 4)}')

        p.set_xdata(wa.x_t(t[i]))
        return psi_gh, psi_rk, psi_cn, p

    plt.legend()
    anim = FuncAnimation(fig, __update, init_func=__init,
                         frames=len(t), interval=100, blit=True)

    if save:
        anim.save('./wave_packet.gif', writer=PillowWriter(fps=FPS),
                  progress_callback=lambda i, n: print(f'Saving frame {i + 1} of {n}'))
        sys.exit(0)
    else:
        plt.show()


if __name__ == "__main__":
    main(save=False)
