import numpy as np
import WFAnimation as wfa
import WFTool as wft
from matplotlib import pyplot as plt
from WaveFunction import Potential


def wave_function(function, *args):
    f = getattr(Potential, function)
    x = np.linspace(-2., 2., 500)
    y_real = np.zeros_like(x)
    if function == 'inf_pot':
        a, = args
        for E_i in np.linspace(0., 4., 100):
            for i in range(len(x)):
                y = f(E_i, a, x[i])
                y_real[i] = np.imag(y)
            plt.plot(x, y_real)
            plt.plot(np.array([-3., 0., 0., a, a, 3.]), np.array([6., 6., 0., 0., 6., 6.]),
                     linestyle='dashed', color='black', label=r'$V_0=0$')
            plt.plot(np.array([-3., 3.]), np.array([E_i, E_i]),
                     linestyle='dashed', label=r'$E=$' + str(round(E_i, 2)))
            plt.legend()
            wfa.render_frame(y_label='Im{$\Psi(x,0)$}')
            wfa.save_frame(function)
    elif function == 'wall':
        V_0, d = args
        for E_i in np.linspace(0.5, 2.5, 100):
            for i in range(len(x)):
                y = f(E_i, V_0, d, x[i])
                y_real[i] = np.real(y)
            plt.plot(x, y_real)
            plt.plot(np.array([-3., 0., 0., d, d, 3.]), np.array([0., 0., V_0, V_0, 0., 0.]),
                     linestyle='dashed', color='black', label=r'$V_0=1$')
            plt.plot(np.array([-3., 3.]), np.array([E_i, E_i]),
                     linestyle='dashed', label=r'$E=$' + str(round(E_i, 2)))
            plt.legend()
            wfa.render_frame()
            wfa.save_frame(function)
    elif function == 'step':
        V_0, = args
        for E_i in np.linspace(0.5, 2.5, 100):
            for i in range(len(x)):
                y = f(E_i, V_0, x[i])
                y_real[i] = np.real(y)
            plt.plot(x, y_real)
            plt.plot(np.array([-3., 0., 3.]), np.array([0., V_0, V_0]),
                     drawstyle='steps-post', linestyle='dashed', color='black', label=r'$V_0=1$')
            plt.plot(np.array([-3., 3.]), np.array([E_i, E_i]),
                     linestyle='dashed', label=r'$E=$' + str(round(E_i, 2)))
            plt.legend()
            wfa.render_frame()
            wfa.save_frame(function)
    elif function == 'fin_pot':
        V_0, a = args
        for E_i in np.linspace(0., 30., 100):
            for i in range(len(x)):
                y = f(E_i, V_0, a, x[i])
                y_real[i] = np.real(y)
            plt.plot(x, y_real)
            plt.plot(np.array([-3., 0., 0., a, a, 3.]), np.array([V_0, V_0, 0., 0., V_0, V_0]),
                     linestyle='dashed', color='black', label=r'$V_0=0$')
            plt.plot(np.array([-3., 3.]), np.array([E_i, E_i]),
                     linestyle='dashed', label=r'$E=$' + str(round(E_i, 2)))
            plt.legend()
            wfa.render_frame()
            wfa.save_frame(function)
    elif function == 'harmonic':
        n, omega = args
        V_osc = np.zeros_like(x)
        for n_i in np.linspace(0, n, n + 1):
            H_const, y_const = wft.hermite_const(n_i, omega)
            for i in range(len(x)):
                y, V = f(omega, x[i])
                y_real[i] = y * wft.hermite(n_i, x[i] * y_const) * H_const + n_i
                V_osc[i] = V
            plt.plot(x, y_real)
            plt.plot(x, V_osc,
                     linestyle='dashed', color='black')
            wfa.render_frame(y_limit=[-2, 10])
            wfa.save_frame(function)
    # wfa.save_anim(function)
