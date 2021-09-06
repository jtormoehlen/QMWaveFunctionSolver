import sys
import numpy as np
from WFPlot import wave_function

if __name__ == '__main__':

    # wave_function('step', 4.)
    # wave_function('wall', 9., .1)
    wave_function('wave_packet', 4., 1.5, 2)

    sys.exit(0)
