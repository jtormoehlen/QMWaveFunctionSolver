import sys
import numpy as np
from WFPlot import wave_function

if __name__ == '__main__':

    # wave_function('inf_pot', 1.5)
    # wave_function('step', 4.)
    # wave_function('wall', 9., .1)
    # wave_function('fin_pot', 4., 1.)
    wave_function('harmonic', 2. * np.pi)

    sys.exit(0)
