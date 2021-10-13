import sys

import numpy as np
from matplotlib import animation, pyplot as plt

fig = plt.figure(figsize=(6, 4))
ax = plt.subplot(1, 1, 1)
ax.set_xlim(-100, 100)
ax.set_ylim(-2, 2)
title = ax.set_title('')
line_psi, = ax.plot([], [], label=r'Re$\{ \psi(x,t) \}$')
line_prob, = ax.plot([], [], label=r'$|\psi(x,t)|^2$')
plt.legend(loc=4, fontsize=8, fancybox=False)


def run(x, t, PSI, close=True):
    def animate(i):
        line_psi.set_data(x, np.real(PSI[:, i]))
        return line_psi,

    anim = animation.FuncAnimation(fig, animate,
                                   frames=len(t), interval=200, blit=True)

    print('Generating GIF')
    anim.save('img/PSI_CN.gif', writer='imagemagick', fps=15, dpi=100, extra_args=['-layers Optimize'])
    sys.exit(0) if close else 0


def show(x, t, PSI):
    # norm = np.sum(np.abs(PSI[:, t])**2)
    norm = 1.0
    line_psi.set_data(x, np.real(PSI[:, t] / norm))
    line_prob.set_data(x, np.abs(PSI[:, t])**2 / norm)
    plt.show()
