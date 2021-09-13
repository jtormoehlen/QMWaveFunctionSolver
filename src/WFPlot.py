import numpy as np
import WFAnimation as wfa
from matplotlib import pyplot as plt
from WaveFunction import Potential


def wave_function(function, *args):
    f = getattr(Potential, function)
    x = np.linspace(-10., 10., 500)
    y_real = np.zeros_like(x)
    y_abs = np.zeros_like(x)
    if function == 'wall':
        V_0, d = args
        x_min = -2 - d
        x_max = 2 + d
        for E_n in np.linspace(0.1, 8., 100):
            for i in range(len(x)):
                y = f(E_n, V_0, d, x[i])
                y_real[i] = np.real(y)
            plt.plot(x, y_real)
            plt.plot(np.array([x_min, -d, -d, d, d, x_max]), np.array([0., 0., V_0, V_0, 0., 0.]),
                     linestyle='dashed', color='black', label=r'$V_0=$' + str(V_0))
            plt.plot(np.array([x_min, x_max]), np.array([E_n, E_n]),
                     linestyle='dashed', label=r'$E=$' + str(round(E_n, 2)))
            plt.legend()
            wfa.render_frame(y_limit=[-2, 10], x_limit=[x_min, x_max])
            wfa.save_frame(function)
    if function == 'step':
        V_0, = args
        for E_n in np.linspace(0.1, 8., 100):
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
    if function == 'step_wave_packet':
        E_0, V_0 = args
        classical = getattr(Potential, 'classical')
        for t in np.linspace(0., 20., 100):
            for i in range(len(x)):
                y = f(E_0, V_0, x[i], t)
                y_real[i] = np.real(y)
                y_abs[i] = np.abs(y ** 2)
            # plt.plot(x, y_real, label=r'$t=$' + str(round(t, 2)))
            plt.plot(x, y_abs, label=r'$t=$' + str(round(t, 2)))
            plt.plot(np.array([-10., 0., 10.]), np.array([0., V_0, V_0]),
                     drawstyle='steps-post', linestyle='dashed', color='black', label=r'$V_0=$' + str(V_0))
            plt.plot(np.array([-10., 10.]), np.array([E_0, E_0]),
                     linestyle='dashed', label=r'$E=$' + str(round(E_0, 5)))
            x_classical = classical(E_0, V_0, t)
            plt.plot(x_classical, 0., 'o')
            plt.legend()
            wfa.render_frame(y_limit=[-2, 2], x_limit=[-10, 10])
            wfa.save_frame(function)
    if function == 'wall_wave_packet':
        E_0, V_0, a = args
        for t in np.linspace(0., 20., 100):
            for i in range(len(x)):
                y = f(E_0, V_0, a, x[i], t)
                y_real[i] = np.real(y)
                y_abs[i] = np.abs(y ** 2)
            plt.plot(x, y_real, label=r'$t=$' + str(round(t, 2)))
            # plt.plot(x, y_abs, label=r'$t=$' + str(round(t, 2)))
            plt.plot(np.array([-10, -a, -a, a, a, 10]), np.array([0., 0., V_0, V_0, 0., 0.]),
                     linestyle='dashed', color='black', label=r'$V_0=$' + str(V_0))
            plt.plot(np.array([-10., 10.]), np.array([E_0, E_0]),
                     linestyle='dashed', label=r'$E=$' + str(round(E_0, 5)))
            plt.legend()
            wfa.render_frame(y_limit=[-1.0, 1.5], x_limit=[-10, 10], y_label='$|\Psi(x,t)|^2$')
            wfa.save_frame(function)
    wfa.save_anim(function)
