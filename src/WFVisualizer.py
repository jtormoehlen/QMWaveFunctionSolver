import matplotlib.pyplot as plt
import numpy as np

from WaveFunction import potential_step, potential_wall
from WFAnimation import save_frame, render_anim


if __name__ == '__main__':
    x = np.linspace(-2., 2., 1001)
    y_real = np.zeros_like(x)
    y_real_2 = np.zeros_like(x)
    y_real_3 = np.zeros_like(x)
    y_real_4 = np.zeros_like(x)
    y_imag = np.zeros_like(x)
    y_abs2 = np.zeros_like(x)

    E = 1.
    V_0 = 1.25
    for t_i in np.linspace(0., 2. * np.pi, 100):
        for i in range(len(x)):
            y_real[i] = np.real(potential_step(E, V_0, x[i], t_i))
            # y_imag[i] = np.imag(potential_step(E, V_0, x[i]))
            # y_abs2[i] = np.abs(potential_step(E, V_0, x[i])) ** 2
            # y_real_2[i] = np.real(potential_step(E, V_0, x[i], t=.25))
            # y_real_3[i] = np.real(potential_step(E, V_0, x[i], t=.5))
            # y_real_4[i] = np.real(potential_step(E, V_0, x[i], t=.75))
        plt.plot(x, y_real)
        save_frame('pot_step')
    render_anim('pot_step')


    # plt.plot(x, y_real)
    # plt.plot(x, y_imag, linestyle='dashed')
    # plt.plot(x, y_abs2)
    # plt.plot(x, y_real_2)
    # plt.plot(x, y_real_3)
    # plt.plot(x, y_real_4)
