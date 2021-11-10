import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as ani
import WaveFunction as wf
import WFNumeric as wfn

x = wf.x
dx = wf.dx
t = wfn.t_N

fig = plt.figure(figsize=(6, 4))
ax = plt.subplot(1, 1, 1)
ax.set_xlim(min(x), max(x))
ax.set_ylim(-0.01, 0.1)
ax.set_xlabel(r'$x/$a')
ax.set_ylabel(r'$|\psi(x,t)|^2/$a$^{-1}$')

ax.plot(x, wf.V(x), '--k')

an = wf.psi(x, 0)
psi_an, = ax.plot(x, np.abs(an) ** 2, label='analytical/GaussQuad')
norm_an = wf.norm(an)

cn = wfn.PsiCN()
psi_cn, = ax.plot(x, np.abs(cn.get()) ** 2, label='Crank-Nicolson')
norm_cn = wf.norm(cn.get())

rk = wfn.PsiIVP().psi_ivp()
psi_rk, = ax.plot(x, np.abs(rk.y[:, 0]) ** 2, label='Runge-Kutta')
norm_rk = wf.norm(rk.y[:, 0])


def update(i):
    psi_an.set_ydata(np.abs(wf.psi(x, t[i])) ** 2 / norm_an)

    psi_rk.set_ydata(np.abs(rk.y[:, i]) ** 2 / norm_rk)

    cn.step(i)
    psi_cn.set_ydata(np.abs(cn.get()) ** 2 / norm_cn)
    return psi_an, psi_rk, psi_cn


plt.legend()
plt.tight_layout()
anim = ani.FuncAnimation(fig, update, frames=len(t), interval=100, blit=True)
plt.show()
# anim.save('img/wall.gif', writer='imagemagick', fps=10, dpi=100, extra_args=['-layers Optimize'])
# sys.exit(0)
