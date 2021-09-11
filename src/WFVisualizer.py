import sys
import numpy as np
from WFPlot import wave_function

if __name__ == '__main__':

    # wave_function('step', 4.)
    # wave_function('wall', 9., .1)
    # wave_function('step_wave_packet', 0.5, 0.499)
    wave_function('wall_wave_packet', 0.5, 2.0, 0.05)

    sys.exit(0)
