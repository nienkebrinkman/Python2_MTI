import numpy as np
from obspy.core.stream import Stream
from obspy.core.trace import Trace

class Forward_problem:
    def Solve_forward(self, G, moment):
        # Forward model:
        data = np.matmul(G, moment)
        return data


