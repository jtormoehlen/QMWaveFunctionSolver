import numpy as np
import WFAnimation as wfa
from matplotlib import pyplot as plt
from WaveFunction import Potential


def wave_function(function, *args):
    f = getattr(Potential, function)
    x = np.linspace(-5., 5., 500)
    y_real = np.zeros_like(x)
    if function == 'wall':
        V_0, d = args
        x_min = -2
        x_max = d + 2
        for E_n in np.linspace(0., 8., 100):
            for i in range(len(x)):
                y = f(E_n, V_0, d, x[i])
                y_real[i] = np.real(y)
            plt.plot(x, y_real)
            plt.plot(np.array([x_min-1, 0., 0., d, d, x_max+1]), np.array([0., 0., V_0, V_0, 0., 0.]),
                     linestyle='dashed', color='black', label=r'$V_0=$' + str(V_0))
            plt.plot(np.array([x_min-1, x_max+1]), np.array([E_n, E_n]),
                     linestyle='dashed', label=r'$E=$' + str(round(E_n, 2)))
            plt.legend()
            wfa.render_frame(y_limit=[-2, 10], x_limit=[x_min, x_max])
            wfa.save_frame(function)
    if function == 'step':
        V_0, = args
        for E_n in np.linspace(0., 8., 100):
            for i in range(len(x)):
                y = f(E_n, V_0, x[i])
                y_real[i] = np.real(y)
            plt.plot(x, y_real)
            plt.plot(np.array([-3., 0., 3.]), np.array([0., V_0, V_0]),
                     drawstyle='steps-post', linestyle='dashed', color='black', label=r'$V_0=$' + str(V_0))
            plt.plot(np.array([-3., 3.]), np.array([E_n, E_n]),
                     linestyle='dashed', label=r'$E=$' + str(round(E_n, 2)))
            plt.legend()
            wfa.render_frame(y_limit=[-2, 10])
            wfa.save_frame(function)
    if function == 'wave_packet':
        p_0, V_0, N, = args
        for t in np.linspace(0., 1., 100):
            for i in range(len(x)):
                y = f(p_0, V_0, N, x[i], t)
                y_real[i] = np.real(y)
            plt.plot(x, y_real, label=r'$t=$' + str(round(t, 2)))
            plt.legend()
            wfa.render_frame(y_limit=[-2, 10])
            wfa.save_frame(function)
    wfa.save_anim(function)
