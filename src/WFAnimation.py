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
psi_an, = ax.plot(x, np.abs(an) ** 2, label='Analytical')
norm_an = wf.norm(an)
cn = wfn.PsiCN()
psi_cn, = ax.plot(x, np.abs(cn.evolve(0)) ** 2, label='Crank-Nicolson')
norm_cn = wf.norm(cn.evolve(0))
rk = wfn.PsiRK().solve()
psi_rk, = ax.plot(x, np.abs(rk.y[:, 0]) ** 2, label='Runge-Kutta')
norm_rk = wf.norm(rk.y[:, 0])


def init():
    ax.plot(x, wf.V(x), '--k')
    return psi_an, psi_rk, psi_cn


def update(i):
    psi_an.set_ydata(np.abs(wfa.psi(x, t[i], wfa.psi_x)) ** 2 / norm_an)
    psi_rk.set_ydata(np.abs(rk.y[:, i]) ** 2 / norm_rk)
    psi_cn.set_ydata(np.abs(cn.evolve(i)) ** 2 / norm_cn)
    return psi_an, psi_rk, psi_cn


plt.legend()
plt.tight_layout()
anim = ani.FuncAnimation(fig, update, init_func=init,
                         frames=len(t), interval=100, blit=True)

wf.param_info()
wfa.info(wfa.psi(x, wf.t_col(), wfa.psi_x))
wfn.PsiRK().info(rk.y[:, wfn.t_N.size - 1])
cn.info()
if False:
    anim.save('img/wall.gif', writer='imagemagick',
              fps=5, dpi=100, extra_args=['-layers Optimize'])
    sys.exit(0)
else:
    plt.show()
