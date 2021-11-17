import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as ani
import WaveFunction as wf
import WFNumeric as wfn
import WFAnalytic as wfa

x = wf.x
dx = wf.dx
t = wfn.t_N

fig = plt.figure(figsize=(6, 4))
ax = plt.subplot(1, 1, 1)
ax.set_xlim(min(x), max(x))
ax.set_ylim(-0.01, 0.1)
ax.set_xlabel(r'Position $x/$a')
ax.set_ylabel(r'Probability density $|\psi(x,t)|^2/$a$^{-1}$')

an = wfa.psi(x, 0, wfa.psi_x)
psi_an, = ax.plot(x, np.abs(an) ** 2, label='Gauss-Hermite')
norm_an = wf.norm(an)

rk = wfn.PsiRK().solve()
psi_rk, = ax.plot(x, np.abs(rk.y[:, 0]) ** 2, label='Runge-Kutta')
norm_rk = wf.norm(rk.y[:, 0])

cn = wfn.PsiCN()
psi_cn, = ax.plot(x, np.abs(cn.evolve(0)) ** 2, label='Crank-Nicolson')
norm_cn = wf.norm(cn.evolve(0))

prob_text = ax.text(-90, 0.07, '', fontsize=12)


def init():
    ax.plot(x, wf.V(x), '--k')
    return psi_an, psi_rk, psi_cn, prob_text,


def update(i):
    psi2_an = np.abs(wfa.psi(x, t[i], wfa.psi_x)) ** 2 / norm_an
    psi_an.set_ydata(psi2_an)

    psi2_rk = np.abs(rk.y[:, i]) ** 2 / norm_rk
    psi_rk.set_ydata(psi2_rk)

    psi2_cn = np.abs(cn.evolve(i)) ** 2 / norm_cn
    psi_cn.set_ydata(psi2_cn)

    prob_text.set_text(r'$P_{GH}=$' + f'{round(wf.prob(psi2_an), 4)}\n'
                       r'$P_{RK}=$' + f'{round(wf.prob(psi2_rk), 4)}\n'
                       r'$P_{CN}=$' + f'{round(wf.prob(psi2_cn), 4)}')
    return psi_an, psi_rk, psi_cn, prob_text,


plt.legend()
plt.tight_layout()
anim = ani.FuncAnimation(fig, update, init_func=init,
                         frames=len(t), interval=100, blit=True)

wf.param_info()
wfa.info(wfa.psi(x, wf.t_col(), wfa.psi_x))
wfn.PsiRK().info(rk.y[:, wfn.t_N.size - 1])
cn.info()
if False:
    anim.save('img/barrier90.gif', writer='imagemagick',
              fps=10, dpi=100, extra_args=['-layers Optimize'],
              progress_callback=lambda i, n: print(f'Saving frame {i+1} of {n}'))

    sys.exit(0)
else:
    plt.show()
