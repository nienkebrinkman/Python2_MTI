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

    def get_CC(self,data_obs,data_syn,dt):
        cc_array= np.correlate(data_obs,data_syn,'full')
        time_shift_array = np.array(range(-len(data_obs)+1,len(data_obs)))

        # The maximum correlation corresponds to the best fit:
        max_cc = np.max(cc_array) * dt
        time_shift = time_shift_array[np.argmax(cc_array,axis=0)]

        d_syn_new = np.zeros_like(data_syn)
        d_obs_new = np.zeros_like(data_obs)
        if time_shift < 0:
            d_syn_new[:time_shift] = data_syn[-time_shift:]
            d_obs_new[:time_shift] = data_obs[-time_shift:]
        elif time_shift == 0:
            d_syn_new[:] = data_syn[:]
            d_obs_new[:] = data_obs[:]
        else:
            d_syn_new[time_shift:] = data_syn[:-time_shift]
            d_obs_new[time_shift:] = data_obs[:-time_shift]

        norm_syn= np.sqrt(np.sum((d_syn_new**2)*dt))
        norm_obs= np.sqrt(np.sum((d_obs_new**2)*dt))

        CC = (max_cc) / (norm_syn * norm_obs)
        return CC,time_shift

