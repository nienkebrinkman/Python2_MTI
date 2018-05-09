import numpy as np

class Misfit:
    def get_RMS(self, data_obs, data_syn):
        N = data_syn.__len__()  # Normalization factor
        RMS = np.sqrt(np.sum(data_obs - data_syn) ** 2 / N)
        return RMS

    def get_xi(self, data_obs, data_syn, var):

        likelihood = np.matmul((data_obs - data_syn).T, (data_obs - data_syn)) / (2 * (var ** 2))
        return likelihood

    def norm(self, data_obs, data_syn):
        norm = np.linalg.norm(data_obs - data_syn)
        return norm