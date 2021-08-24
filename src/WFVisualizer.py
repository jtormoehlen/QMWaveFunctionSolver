import sys

import numpy as np

from WFPlot import wave_function
from src.WaveFunction import Potential

if __name__ == '__main__':

    # wave_function('inf_pot', 1.5)
    # wave_function('step', 1.)
    # wave_function('wall', 4., .1)
    # wave_function('fin_pot', 16., 1.)
    wave_function('harmonic', 5, 2. * np.pi)

    sys.exit(0)
