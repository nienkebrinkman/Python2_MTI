import numpy as np
import obspy.signal.cross_correlation as cc
import matplotlib.pylab as plt

class Misfit:
    def get_RMS(self, data_obs, data_syn):
        N = data_syn.__len__()  # Normalization factor
        RMS = np.sqrt(np.sum(data_obs - data_syn) ** 2 / N)
        return RMS

    def get_xi_stream(self,trace_obs,trace_syn,var,components):
        cc_array=cc.xcorr_3c(st1=trace_obs,st2=trace_syn,shift_len=(len(trace_obs[0])-1)/2,components=components,full_xcorr=True)
        # TODO 1 choose max value and the corresponding index shift
        # TODO 2 Shift the synthetic data with this index shift
        # TODO 3 Calculate the misfit

        misfit = np.matmul((data_obs - data_syn).T, (data_obs - data_syn)) / (2 * (var ** 2))
        return misfit


    def get_xi(self, data_obs, data_syn, var,dt):
        cc_array= np.correlate(data_obs,data_syn,'full')
        time_shift_array = np.array(range(-len(data_obs)+1,len(data_obs)))
        # The maximum correlation corresponds to the best fit:
        max_cc = np.max(cc_array)
        time_shift = time_shift_array[np.argmax(cc_array,axis=0)]
        # Now shift d_syn with the time shift corresponding to the maximum Cross-Correlation:
        data_syn_shift = np.roll(data_syn, time_shift)

        plt.plot(data_syn, alpha = 0.5,label='Original Synthetic data')
        plt.plot(data_syn_shift,alpha= 0.5,color='r',label='Shifted synthetic data')
        plt.xlabel('Time samples: %.3f[sec/sample]' %dt)
        plt.ylabel('Displacement [m]')
        plt.legend()
        # plt.plot(data_obs, linestyle=':')
        plt.savefig('./Plots/shifts/d_syn_shift_%i.pdf' %time_shift)
        plt.close()
        misfit = np.matmul((data_obs - data_syn_shift).T, (data_obs - data_syn_shift)) / (2 * (var ** 2))

        return misfit

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
        D = 1 - CC
        return D,time_shift

