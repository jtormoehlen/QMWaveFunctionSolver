import numpy as np
import WaveFunction as wf
from matplotlib import pyplot as plt
from matplotlib import animation as ani


def animate(x, animated=True, t=0):
    fig = plt.figure(figsize=(6, 4))
    ax = plt.subplot(1, 1, 1)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-2, 2)
    ax.set_xlabel(r'$x$')
    psi, = ax.plot(x, np.real(wf.psi_N(x, t)), label=r'Re{$\psi(x,t)$}')
    prob, = ax.plot(x, np.abs(wf.psi_N(x, t))**2, label=r'$|\psi(x,t)|^2$')
    V_x = ax.plot(x, wf.V(x) / 10, '--k')
    # particle, = ax.plot([-10, -10], [-2, 2], '--g')
    # prob = ax.text(0.05, 0.90, '', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

    def update(i):
        dt = 0.01 * i
        psi_i = wf.psi_N(x, dt)
        psi.set_ydata(np.real(psi_i))
        prob.set_ydata(np.abs(psi_i)**2)
        # prob.set_text('Tun.-Wahr.: ' + str(round(wf.prob(), 4)))
        # particle.set_xdata([wf.x_p(dt), wf.x_p(dt)])
        return psi, prob,

    if animated:
        ani.FuncAnimation(fig, update, interval=50, blit=True)
    else:
        plt.legend()

    plt.show()
