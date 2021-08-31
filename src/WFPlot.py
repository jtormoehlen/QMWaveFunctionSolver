import numpy as np
import WFAnimation as wfa
import WFTool as wft
from matplotlib import pyplot as plt
from WaveFunction import Potential


def wave_function(function, *args):
    f = getattr(Potential, function)
    x = np.linspace(-5., 5., 500)
    y_real = np.zeros_like(x)
    if function == 'inf_pot':
        a, = args
        y_max = 10
        x_min = -2
        x_max = a + 2
        for E_n in np.linspace(0., 8., 100):
            for i in range(len(x)):
                y = f(E_n, a, x[i])
                y_real[i] = y
            plt.plot(x, y_real)
            plt.plot(np.array([x_min-1, 0., 0., a, a, x_max+1]),
                     np.array([y_max+1, y_max+1, 0., 0., y_max+1, y_max+1]),
                     linestyle='dashed', color='black', label=r'$V \rightarrow \infty$')
            plt.plot(np.array([x_min-1, x_max+1]), np.array([E_n, E_n]),
                     linestyle='dashed', label=r'$E=$' + str(round(E_n, 2)))
            plt.legend()
            wfa.render_frame(y_label='Im{$\Psi(x,0)$}', y_limit=[-2, y_max], x_limit=[x_min, x_max])
            wfa.save_frame(function)
    elif function == 'wall':
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
    elif function == 'step':
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
    elif function == 'fin_pot':
        V_0, a = args
        x_min = -2
        x_max = a + 2
        for E_n in np.linspace(0., 8., 100):
            for i in range(len(x)):
                y = f(E_n, V_0, a, x[i])
                y_real[i] = np.real(y)
            plt.plot(x, y_real)
            plt.plot(np.array([x_min-1, 0., 0., a, a, x_max+1]), np.array([V_0, V_0, 0., 0., V_0, V_0]),
                     linestyle='dashed', color='black', label=r'$V_0=$' + str(V_0))
            plt.plot(np.array([x_min-1, x_max+1]), np.array([E_n, E_n]),
                     linestyle='dashed', label=r'$E=$' + str(round(E_n, 2)))
            plt.legend()
            wfa.render_frame(y_limit=[-2, 10], x_limit=[x_min, x_max])
            wfa.save_frame(function)
    elif function == 'harmonic':
        omega, = args
        V_osc = np.zeros_like(x)
        for E_n in np.linspace(omega / 2., 30., 100):
            n = np.floor((E_n - (omega / 2.)) / omega)
            if n < 0:
                n = 0
            E_0 = omega * (n + 1. / 2.)
            H_const, y_const = wft.hermite_const(n, omega)
            for i in range(len(x)):
                y, V = f(omega, x[i])
                y_real[i] = y * wft.hermite(n, x[i] * y_const) * H_const + E_0
                V_osc[i] = V
            plt.plot(x, y_real)
            plt.plot(x, V_osc,
                     linestyle='dashed', color='black', label=r'$\omega_0=$' + str(omega / np.pi) + r'$\pi$')
            plt.plot(np.array([-3., 3.]), np.array([E_n, E_n]),
                     linestyle='dashed', label=r'$E=$' + str(round(E_n, 2)))
            plt.legend()
            wfa.render_frame(y_limit=[-2, 40])
            wfa.save_frame(function)
    wfa.save_anim(function)
