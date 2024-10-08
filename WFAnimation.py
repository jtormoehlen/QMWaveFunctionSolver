import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from lib.WFNumeric import RKSolver, CNSolver
from lib.WFAnalytic import GHSolver
# from expl.WFBarrier import Barrier
from expl.WFPot import Pot

plt.style.use('./lib/figstyle.mpstyle')
FPS = 10 # frames per second

# start animation
# save: generate .gif in /img if true
def main(save):
    # compute |psi(x,0)|^2 by GH; |psi(x,t)|^2 by RK and CN
    # plot |psi(x,0)|^2 as [Line2D,...]
    # compute norm |psi(x,0)|^2
    fig, ax = plt.subplots()

    model = Pot("Pot")
    # model = Barrier("Barrier")
    x = model.x # spatial coords

    gh = GHSolver(model)
    model.params(gh.psi(-1))
    rk = RKSolver(model)
    cn = CNSolver(model)
    pgh, = ax.plot(x, np.abs(gh.psi(0))**2, label='Gauss-Hermite')
    prk, = ax.plot(x, np.abs(rk.psi(0))**2, label='Runge-Kutta')
    pcn, = ax.plot(x, np.abs(cn.psi(0))**2, label='Crank-Nicolson')
    # classical p position x(0)
    px, = ax.plot(model.xpos(0), 0.0, 'ok') 

    # plot potential V(x)
    # return: wave packets as [[Line2D,...],[Line2D,...],...]
    def init():
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(-0.1*max(np.abs(gh.psi(0))**2),
                    max(np.abs(gh.psi(0))**2)+0.1*max(np.abs(gh.psi(0))**2))
        ax.set_xlabel('Position $x/$a')
        ax.set_ylabel('Probability density $|\psi(x,t)|^2/$a$^{-1}$')
        ax.plot(x, model.V(x), '--k')
        return pgh, prk, pcn, px,

    # 1.) compute normalized |psi(x,i)|^2 by GH, RK and CN
    # 2.) plot |psi(x,i)|^2 as [Line2D,...]
    # i: time (frame) in (1,2,...,|tn|)
    # return: wave packets as [[Line2D,...],[Line2D,...],...]
    def update(i):
        pgh.set_ydata(np.abs(gh.psi(i))**2)
        prk.set_ydata(np.abs(rk.psi(i))**2)
        pcn.set_ydata(np.abs(cn.psi(i))**2)
        px.set_xdata(model.xpos(i))
        return pgh, prk, pcn, px,

    plt.legend()
    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(model.t), interval=100, blit=True)

    if save:
        anim.save(f'./img/{model.name}.gif', writer=PillowWriter(fps=FPS),
                  progress_callback=lambda i, n: print(f'Saving frame {i+1} of {n}'))
        sys.exit(0)
    else:
        plt.show()

if __name__ == '__main__':
    main(save=True)