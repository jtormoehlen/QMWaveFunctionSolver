import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as ani

import WaveFunction as wf
import WFNumerical as wfn

x_max = wf.x_max
x = wf.x
dx = wf.dx

fig = plt.figure(figsize=(6, 4))
ax = plt.subplot(1, 1, 1)
ax.set_xlim(-x_max, x_max)
ax.set_ylim(-0.1, 0.5)
ax.set_xlabel(r'$x/$a')
ax.set_ylabel(r'$|\psi(x,t)|^2/$a$^{-1}$')
text_analytic = ax.text(0.1, 0.8, "",
                        bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                        transform=ax.transAxes, ha="left")
text_numeric = ax.text(0.1, 0.6, "",
                       bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
                       transform=ax.transAxes, ha="left")
wf.info()

psi_analytic, = ax.plot(x, np.abs(wf.psi(x, 0)) ** 2, label='analytical')
sol = wfn.psi_ivp_solve()
psi_numerical, = ax.plot(x, np.abs(sol.y[:, 0]) ** 2, label='numerical')

plt.tight_layout()
ax.plot(x, wf.V(x), '--k')

limiter = int(len(x) / 2)
norm_analytic = np.sum(np.abs(wf.psi(x, 0)[:limiter]) ** 2 * dx)
norm_numeric = np.sum(np.abs(sol.y[:limiter, 0]) ** 2 * dx)

plt.legend()


def update(i):
    psi_analytic.set_ydata((np.abs(wf.psi(x, sol.t[i])) ** 2) / norm_analytic)
    psi_numerical.set_ydata((np.abs(sol.y[:, i]) ** 2) / norm_numeric)
    reflect_analytic = np.sum(np.abs(wf.psi(x, sol.t[i])[:limiter]) ** 2 * dx)
    trans_analytic = np.sum(np.abs(wf.psi(x, sol.t[i])[limiter:]) ** 2 * dx)
    text_analytic.set_text(r'$R_a = $ {0:1.3f}'.format(reflect_analytic / norm_analytic) +
                           r', $T_a = $ {0:1.3f}'.format(trans_analytic / norm_analytic))
    reflect_numeric = np.sum(np.abs(sol.y[:limiter, i]) ** 2 * dx)
    trans_numeric = np.sum(np.abs(sol.y[limiter:, i]) ** 2 * dx)
    text_numeric.set_text(r'$R_n = $ {0:1.3f}'.format(reflect_numeric / norm_numeric) +
                          r', $T_n = $ {0:1.3f}'.format(trans_numeric / norm_numeric))
    return psi_analytic, psi_numerical


anim = ani.FuncAnimation(fig, update, frames=len(sol.t), interval=100, blit=False)
plt.show()
# anim.save('img/wall.gif', writer='imagemagick', fps=10, dpi=100, extra_args=['-layers Optimize'])
# sys.exit(0)
