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
        time_shift = np.array([], dtype=int)
        # S - correlations:
        for i in range(len(s_obs)):
            cc_obspy = cc.correlate(s_obs[i].data, s_syn[i].data, int(0.25* len(s_obs[i].data)))
            shift, CC_s = cc.xcorr_max(cc_obspy)

            s_syn_shift = self.shift(s_syn[i].data, -shift)
            time_shift = np.append(time_shift, shift)

            # d_obs_mean = np.mean(s_obs[i].data)
            # var_array = var * d_obs_mean

            var_array = np.var(s_obs[i].data)
            # var_array = var**2

            misfit = np.append(misfit, np.matmul((s_obs[i].data - s_syn_shift).T, (s_obs[i].data - s_syn_shift)) / (
                2 * (var_array)))
            # time = -time_shift * dt  # Relatively, the s_wave arrives now time later or earlier than it originally did

            # plt.plot(s_syn_shift,label='s_shifted')
            # plt.plot(s_syn[i], label='s_syn')
            # plt.plot(s_obs[i], label='s_obs')
            # plt.legend()
            # plt.show()
            # plt.close()
        # P- correlation
        for i in range(len(p_obs)):
            cc_obspy = cc.correlate(s_obs[i].data, s_syn[i].data, int(0.25 * len(p_obs[i].data)))
            shift, CC_p = cc.xcorr_max(cc_obspy)

            p_syn_shift = self.shift(p_syn[i].data, -shift)
            time_shift = np.append(time_shift, shift)

            # d_obs_mean = np.mean(p_obs[i].data)
            # var_array = var * d_obs_mean
            var_array = np.var(p_obs[i].data)

            misfit = np.append(misfit, np.matmul((p_obs[i].data - p_syn_shift).T, (p_obs[i].data - p_syn_shift)) / (
                2 * (var_array)))
            # time = -time_shift + len(s_obs)] * dt  # Relatively, the s_wave arrives now time later or earlier than it originally did

            # plt.plot(p_syn_shift, label='p_shifted')
            # plt.plot(p_syn[i], label='p_syn')
            # plt.plot(p_obs[i], label='p_obs')
            # plt.legend()
            # # plt.show()
            # plt.close()
        sum_misfit = np.sum(misfit)
        return sum_misfit, time_shift

    def CC_stream(self, p_obs, p_syn, s_obs, s_syn, or_time):
        dt = s_obs[0].meta.delta
        misfit = np.array([])
        misfit_obs = np.array([])
        time_shift = np.array([], dtype=int)
        # S - correlations:
        for i in range(len(s_obs)):
            cc_obspy = cc.correlate(s_obs[i].data, s_syn[i].data, int(0.25*len(s_obs[i].data)))
            shift, CC_s = cc.xcorr_max(cc_obspy, abs_max=False)

            s_syn_shift_obspy = self.shift(s_syn[i].data, -shift)

            D_s = 1 - CC_s  # Decorrelation
            time_shift = np.append(time_shift, shift)
            misfit = np.append(misfit, ((CC_s - 0.95) ** 2) / (2 * (0.1) ** 2))# + np.abs(shift))

            # misfit = np.append(misfit,((CC - 0.95) ** 2) / (2 * (0.1) ** 2))

            # plt.plot(s_syn_shift_obspy,label='s_syn_shifted_obspy')
            # plt.plot(s_syn[i],label = 's_syn')
            # plt.plot(s_obs[i],label = 's_obs')
            # plt.legend()
            # plt.show()
            # plt.close()
        # P- correlation
        for i in range(len(p_obs)):
            cc_obspy = cc.correlate(p_obs[i].data, p_syn[i].data, int( 0.25*len(p_obs[i].data)))
            shift, CC_p = cc.xcorr_max(cc_obspy ,abs_max=False)

            p_syn_shift_obspy = self.shift(p_syn[i].data, -shift)

            D_p = 1 - CC_p  # Decorrelation
            time_shift = np.append(time_shift, shift)
            misfit = np.append(misfit, ((CC_p - 0.95) ** 2) / (2 * (0.1) ** 2)) #+ np.abs(shift))

            # misfit = np.append(misfit,((CC - 0.95) ** 2) / (2 * (0.1) ** 2))

            # plt.plot(p_syn_shift_obspy,label='p_syn_shifted_obspy')
            # # plt.plot(p_syn[i],label = 'p_syn')
            # plt.plot(p_obs[i],label = 'p_obs')
            # plt.legend()
            # plt.show()
            # plt.close()
        sum_misfit = np.sum(misfit)
        return misfit, time_shift

    def shift(self, np_array, time_shift):
        new_array = np.zeros_like(np_array)
        if time_shift < 0:
            new_array[-time_shift:] = np_array[:time_shift]
        elif time_shift == 0:
            new_array[:] = np_array[:]
        else:
            new_array[:-time_shift] = np_array[time_shift:]
        return new_array

    def SW_CC(self, SW_env_obs, SW_env_syn):
        # I suppose that your observed traces and synthetic traces are filtered with same bandwidths in same order!

        R_dict = {}
        misfit = np.array([])
        for i in range((len(SW_env_obs))):
            dt = SW_env_obs[i].meta.delta

            cc_obspy = cc.correlate(SW_env_obs[i].data, SW_env_syn[i].data, int(0.25*len(SW_env_obs[i].data)))
            shift, CC = cc.xcorr_max(cc_obspy)

            SW_syn_shift_obspy = self.shift(SW_env_syn[i].data, -shift)

            D = 1 - CC  # Decorrelation
            misfit = np.append(misfit, ((CC - 0.95) ** 2) / (2 * (0.1) ** 2) + np.abs(shift))
            R_dict.update({'%s' % SW_env_obs.traces[i].stats.channel: {'misfit': misfit[i], 'time_shift': shift}})
        sum_misfit = np.sum(misfit)
        return sum_misfit

    def SW_L2(self, SW_env_obs, SW_env_syn, var):
        misfit = np.array([])
        for i in range(len(SW_env_obs)):
            dt = SW_env_obs[i].meta.delta

            cc_obspy = cc.correlate(SW_env_obs[i].data, SW_env_syn[i].data,int( 0.25*len(SW_env_syn[i].data)))
            shift, CC_s = cc.xcorr_max(cc_obspy)
            # plt.plot(SW_env_obs[i].data,label= 'obs')
            # plt.plot(SW_env_syn[i].data,':',label = 'syn')
            # # # plt.plot(cc_obspy,label = 'CC')
            # plt.legend()
            # plt.show()

            SW_syn_shift = self.shift(SW_env_syn[i].data, -shift)

            # d_obs_mean = np.mean(SW_env_obs[i].data)
            # var_array = var * d_obs_mean

            var_array =  np.var(SW_env_obs[i].data)

            misfit = np.append(misfit,
                               np.matmul((SW_env_obs[i].data - SW_syn_shift).T, (SW_env_obs[i].data - SW_syn_shift)) / (
                                   2 * (var_array )))
            time = -shift * dt
            # plt.plot(SW_syn_shift,label='R_syn_shifted')
            # plt.plot(SW_env_syn[i],label = 'R_syn')
            # plt.plot(SW_env_obs[i],label = 'R_obs')
            # plt.legend()
            # plt.show()
            # plt.close()
        sum_misfit = np.sum(misfit)
        return misfit
