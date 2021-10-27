import sys

import numpy as np
import WaveFunction as wf
from matplotlib import pyplot as plt
from matplotlib import animation as ani


fig = plt.figure(figsize=(6, 4))
ax = plt.subplot(1, 1, 1)
x_max = 50
ax.set_xlim(-x_max, x_max)
# ax.set_ylim(-1, 20)
ax.set_xlabel(r'$x/$a')
ax.set_ylabel(r'$|\psi(x,t)|^2/a^{-1}$')
x = np.linspace(-x_max, x_max, 201)
psi, = ax.plot(x, np.abs(wf.psi(x, 0))**2, label=r'Re{$\psi(x,t)$}')
max = max(np.abs(wf.psi(x, 0))**2)
ax.plot(x, wf.V(x) * max * 2, '--k')
wf.info()


def update(i):
    psi_i = wf.psi(x, i)
    psi.set_ydata(np.abs(psi_i)**2)
    return psi,


anim = ani.FuncAnimation(fig, update, frames=100, interval=100, blit=True)
plt.show()
# anim.save('img/free.gif', writer='imagemagick', fps=10, dpi=100, extra_args=['-layers Optimize'])
# sys.exit(0)
