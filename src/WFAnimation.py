import sys

import numpy as np
from matplotlib import animation, pyplot as plt


fig = plt.figure(figsize=(6, 4))
ax = plt.subplot(1, 1, 1)
ax.set_xlim(-5, 5)
ax.set_ylim(-2, 2)
title = ax.set_title('')
line_V, = ax.plot([], [], 'k--', label=r'$V(x)$')
line_psi, = ax.plot([], [], label=r'$Re\{ \psi \}$')
plt.legend(loc=4, fontsize=8, fancybox=False)


def run(x, V, sol, close=True):
    def init():
        line_V.set_data(x, V * 0.001)
        return line_V,

    def animate(i):
        line_psi.set_data(x, np.real(sol.y[:, i]))
        title.set_text('Time = {0:1.3f}'.format(sol.t[i]))
        return line_psi,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(sol.t), interval=200, blit=True)

    print('Generating GIF')
    anim.save('img/linear.gif', writer='imagemagick', fps=15, dpi=100, extra_args=['-layers Optimize'])
    sys.exit(0) if close else 0
