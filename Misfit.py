import numpy as np
import matplotlib.pylab as plt
import obspy.signal.cross_correlation as cc
import obspy
from obspy.core.stream import Stream
from obspy.core.trace import Trace
import os


class Misfit:
    def __init__(self, directory):
        self.save_dir = directory

    def get_RMS(self, data_obs, data_syn):
        N = data_syn.__len__()  # Normalization factor
        RMS = np.sqrt(np.sum(data_obs - data_syn) ** 2 / N)
        return RMS

    def norm(self, data_obs, data_syn):
        norm = np.linalg.norm(data_obs - data_syn)
        return norm

    def L2_stream(self, p_obs, p_syn, s_obs, s_syn, or_time, var):
        dt = s_obs[0].meta.delta
        misfit = np.array([])
        time_shift = np.array([],dtype=int)
        # S - correlations:
        for i in range(len(s_obs)):
            cc_array = np.correlate(s_obs[i].data, s_syn[i].data, 'same')
            time_shift_array = np.flip(np.array(range(-len(s_obs[i].data) / 2, len(s_obs[i].data) / 2 + 1)), axis=0)

            # The maximum correlation corresponds to the best fit:
            s_corr_max =  np.max(cc_array)
            time_shift = np.append(time_shift, time_shift_array[np.argmax(cc_array, axis=0)])
            s_syn_shift = self.shift(s_syn[i].data,time_shift[i])

            d_obs_mean = np.mean(s_obs[i].data)
            var_array = var * d_obs_mean

            misfit = np.append(misfit,np.matmul((s_obs[i].data - s_syn_shift).T, (s_obs[i].data - s_syn_shift)) / (2 * (var_array ** 2)))
            time = -time_shift[i] * dt  # Relatively, the s_wave arrives now time later or earlier than it originally did

            # plt.plot(s_syn_shift,label='s_shifted')
            # plt.plot(s_syn[i],label = 's_syn')
            # plt.plot(s_obs[i],label = 's_obs')
            # plt.legend()
            # plt.show()
            # plt.close()
        # P- correlation
        for i in range(len(p_obs)):
            cc_array = np.correlate(p_obs[i].data, p_syn[i].data, 'same')
            time_shift_array = np.flip(np.array(range(-len(p_obs[i].data) / 2, len(p_obs[i].data) / 2 + 1)), axis=0)

            # The maximum correlation corresponds to the best fit:
            corr_max = np.max(cc_array)
            time_shift = np.append(time_shift, time_shift_array[np.argmax(cc_array, axis=0)])
            p_syn_shift = self.shift(p_syn[i].data,time_shift[i+len(s_obs)])

            d_obs_mean = np.mean(p_obs[i].data)
            var_array = var * d_obs_mean

            misfit = np.append(misfit, np.matmul((p_obs[i].data - p_syn_shift).T, (p_obs[i].data - p_syn_shift)) / (
            2 * (var_array ** 2)))
            time = -time_shift[i+len(s_obs)] * dt  # Relatively, the s_wave arrives now time later or earlier than it originally did

            # plt.plot(p_syn_shift, label = 'p_shifted')
            # plt.plot(p_syn[i], label = 'p_syn')
            # plt.plot(p_obs[i], label = 'p_obs')
            # plt.legend()
            # # plt.show()
            # plt.close()
        sum_misfit=np.sum(misfit)
        return sum_misfit,time_shift

    def CC_stream(self, p_obs, p_syn, s_obs, s_syn, or_time):
        dt = s_obs[0].meta.delta
        misfit = np.array([])
        time_shift = np.array([],dtype=int)
        # S - correlations:
        for i in range(len(s_obs)):
            cc_array = np.correlate(s_obs[i].data, s_syn[i].data, 'same')
            time_shift_array = np.flip(np.array(range(-len(s_obs[i].data) / 2, len(s_obs[i].data) / 2 + 1)), axis=0)

            # The maximum correlation corresponds to the best fit:
            s_corr_max =  np.max(cc_array)
            time_shift = np.append(time_shift, time_shift_array[np.argmax(cc_array, axis=0)])
            s_syn_shift = self.shift(s_syn[i].data,time_shift[i])
            s_obs_shift = self.shift(s_obs[i].data,time_shift[i])

            norm_syn_old = np.sqrt(np.sum((s_syn_shift ** 2) * dt))
            norm_syn = np.linalg.norm(s_syn_shift)
            norm_obs_old= np.sqrt(np.sum((s_obs_shift ** 2) * dt))
            norm_obs = np.linalg.norm(s_obs_shift)
            # CC = max_cc
            CC_s = (s_corr_max) / (norm_syn * norm_obs)  # Cross-Correlation
            D_s = 1 - CC_s  # Decorrelation
            misfit = np.append(misfit,((CC_s - 1) ** 2) / (2 * (0.1) ** 2)+np.abs(time_shift[i]))
            # misfit = np.append(misfit,((CC - 0.95) ** 2) / (2 * (0.1) ** 2))

            # plt.plot(s_syn_shift,label='s_syn_shifted')
            # plt.plot(s_obs_shift,label = 's_obs_shifted')
            # plt.plot(s_syn[i],label = 's_syn')
            # plt.plot(s_obs[i],label = 's_obs')
            # plt.legend()
            # plt.show()
            # plt.close()
        # P- correlation
        for i in range(len(p_obs)):
            cc_array = np.correlate(p_obs[i].data, p_syn[i].data, 'same')
            time_shift_array = np.flip(np.array(range(-len(p_obs[i].data) / 2, len(p_obs[i].data) / 2 + 1)), axis=0)

            # The maximum correlation corresponds to the best fit:
            p_corr_max =  np.max(cc_array)
            time_shift = np.append(time_shift, time_shift_array[np.argmax(cc_array, axis=0)])
            p_syn_shift = self.shift(p_syn[i].data,time_shift[i+len(s_obs)])
            p_obs_shift = self.shift(p_obs[i].data,time_shift[i+len(s_obs)])

            norm_syn_old = np.sqrt(np.sum((p_syn_shift ** 2) * dt))
            norm_syn = np.linalg.norm(p_syn_shift)
            norm_obs_old = np.sqrt(np.sum((p_obs_shift ** 2) * dt))
            norm_obs = np.linalg.norm(p_obs_shift)
            # CC = max_cc
            CC_p = (p_corr_max) / (norm_syn * norm_obs)  # Cross-Correlation
            D_p = 1 - CC_p  # Decorrelation
            # misfit = np.append(misfit,((CC - 0.95) ** 2) / (2 * (0.1) ** 2))
            misfit = np.append(misfit,((CC_p- 1) ** 2) / (2 * (0.1) ** 2)+np.abs(time_shift[i+len(s_obs)]))

            # plt.plot(p_syn_shift,label = 'p_syn_shifted')
            # plt.plot(p_obs_shift,label = 'p_obs_shifted')
            # plt.plot(p_syn[i],label    = 'p_syn')
            # plt.plot(p_obs[i],label    = 'p_obs')
            # plt.legend()
            # plt.show()
            # plt.close()
        sum_misfit = np.sum(misfit)
        return sum_misfit, time_shift

    def shift(self,np_array,time_shift):
        new_array=np.zeros_like(np_array)
        if time_shift < 0:
            new_array[-time_shift:] = np_array[:time_shift]
        elif time_shift == 0:
            new_array[:] =np_array[:]
        else:
            new_array[:-time_shift] =np_array[time_shift:]
        return new_array

