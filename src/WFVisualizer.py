import matplotlib.pyplot as plt
import numpy as np

from WaveFunction import potential_step, potential_wall

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x = np.linspace(-1., 1., 1001)
    y_real = np.zeros_like(x)
    y_imag = np.zeros_like(x)

    E = 1.
    for i in range(len(x)):
        y_real[i] = np.real(potential_wall(E, x[i]))
        # y_imag[i] = np.imag(potential_step(E, x[i]))

    plt.plot(x, y_real)
    # plt.plot(x, y_imag)
    plt.show()
