import numpy as np

class Forward_problem:
    def __init__(self, PARAMETERS, G, moment):
        self.par = PARAMETERS
        self.moment = moment
        self.G = G

    def Solve_forward(self):
        # Forward model:
        data = np.matmul(self.G, self.moment)
        return data
