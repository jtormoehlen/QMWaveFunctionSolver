import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import WFUtil as wf
import WFNumeric as wn
import WFAnalytic as wa


def main():
    x = wf.x
    t = wn.t_n

    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot(1, 1, 1)

    an = wa.psi(x, 0)
    psi_an, = ax.plot(x, np.abs(an) ** 2, label='Gauss-Hermite')
    norm_an = wf.norm(an)

    rk = wn.RKSolver()
    psi_rk, = ax.plot(x, np.abs(rk.psi(0)) ** 2, label='Runge-Kutta')
    norm_rk = wf.norm(rk.psi(0))

    cn = wn.CNSolver()
    psi_cn, = ax.plot(x, np.abs(cn.psi(0)) ** 2, label='Crank-Nicolson')
    norm_cn = wf.norm(cn.psi(0))

    prob_text = ax.text(wf.x_0, 0.12, '', fontsize=12)
    particle, = ax.plot(wf.x_0, 0., 'ok')

    ax.set_xlim(min(x), max(x))
    ax.set_ylim(-0.1 * max(np.abs(an) ** 2 / norm_an),
                max(np.abs(an) ** 2 / norm_an) + 0.1 * max(np.abs(an) ** 2 / norm_an))
    ax.set_xlabel(r'Position $x/$a')
    ax.set_ylabel(r'Probability density $|\psi(x,t)|^2/$a$^{-1}$')

    def init():
        ax.plot(x, wf.V(x), '--k')
        return psi_an, psi_rk, psi_cn, particle

    def update(i):
        psi2_an = np.abs(wa.psi(x, t[i])) ** 2 / norm_an
        psi_an.set_ydata(psi2_an)

        psi2_rk = np.abs(rk.psi(i)) ** 2 / norm_rk
        psi_rk.set_ydata(psi2_rk)

        psi2_cn = np.abs(cn.psi(i)) ** 2 / norm_cn
        psi_cn.set_ydata(psi2_cn)

        prob_text.set_text(r'$P_{GH}=$' + f'{round(wf.prob(psi2_an), 4)}\n'
                                          r'$P_{RK}=$' + f'{round(wf.prob(psi2_rk), 4)}\n'
                                                         r'$P_{CN}=$' + f'{round(wf.prob(psi2_cn), 4)}')

        particle.set_xdata(wa.x_t(t[i]))
        return psi_an, psi_rk, psi_cn, particle

    plt.legend()
    plt.tight_layout()
    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(t), interval=100, blit=True)

    wf.param_info()
    wa.prob_info(wa.psi(x, t[-1]))
    rk.prob_info()
    cn.prob_info()

    anim.save('./wave_packet.gif', writer=PillowWriter(fps=10),
              progress_callback=lambda i, n: print(f'Saving frame {i + 1} of {n}'))
    sys.exit(0)
    # plt.show()


if __name__ == "__main__":
    main()
