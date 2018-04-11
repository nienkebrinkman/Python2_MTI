import numpy as np

class Inversion_problem:
    def __init__(self, data, G, PARAMETERS):
        self.alpha = PARAMETERS['alpha']
        self.beta = PARAMETERS['beta']
        self.m_ref = PARAMETERS['m_ref']
        self.data = data
        self.G = G

    def Solve_LS(self):
        M = np.linalg.lstsq(self.G, self.data)
        print('Least-square: \n %s' % M[0])
        return M

    def Solve_regularization(self):
        M = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.G.T, self.G)), self.G.T), self.data)
        print('Regularization: \n %s' % M)
        return M

    def Solve_damping(self):
        I = np.eye(self.m_ref.__len__())
        M = np.matmul(np.linalg.inv(np.matmul(self.G.T, self.G) + (I * self.alpha ** 2)),
                      (np.matmul(self.G.T, self.data) + np.matmul((I * self.alpha ** 2), self.m_ref)))
        print('Damping : \n%s' % M)
        return M

    def Solve_damping_smoothing(self):
        I = np.eye(self.m_ref.__len__())
        I[1, :] = 0  # Because m1+m2+m3=0
        trace = np.array([1, 1, 0, 0, 0])
        trace.shape = (1, 5)
        trace_matrix = np.matmul(trace.T, trace)
        M = np.matmul(
            np.linalg.inv(np.matmul(self.G.T, self.G) + (I * self.alpha ** 2) + self.beta ** 2 * trace_matrix),
            (np.matmul(self.G.T, self.data) + np.matmul((I * self.alpha ** 2), self.m_ref)))
        print('Damping & smoothening:\n %s' % M)
        return M

    def Solve_SVD(self):
        ## Solve Singular Value Decomposition (SVD)
        U, s, V = np.linalg.svd(self.G, full_matrices=True)
        s_diag = np.zeros((U.__len__(), V.__len__()))
        s_diag[:V.__len__(), :V.__len__()] = np.diag(s)
        M_SVD = np.matmul(np.matmul(V, np.linalg.pinv(s_diag)), np.matmul(U.T, self.data))
        print('SVD: \n%s' % M_SVD)

        # For regularization more work needs to be done:
        # T_diag = np.diag((s ** 2) / (s ** 2 + self.alpha))
        # M_regul = np.matmul(np.matmul(np.matmul(np.matmul(V, T_diag), np.linalg.pinv(s_diag)), U.T), self.data)
        return M_SVD
