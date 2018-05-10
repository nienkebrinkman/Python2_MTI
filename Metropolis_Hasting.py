import numpy as np
import obspy
import os
import mpi4py as MPI
from mpi4py import MPI
import matplotlib.pyplot as plt
import pylab

from Inversion_problems import Inversion_problem
from Misfit import Misfit
from Source_code import Source_code


class MH_algorithm:
    def __init__(self, PARAMETERS, sampler, db, data, traces):
        self.db = db
        self.par = PARAMETERS
        self.sampler = sampler
        self.d_obs = data
        self.traces = traces

        # Misfit Variance, which should not differ during the MC algorithm:
        d_obs_mean = np.mean(self.d_obs)
        self.var = self.sampler['var_est'] * d_obs_mean

    def model_samples(self):
        epi_sample = np.random.uniform(self.sampler['epi']['range_min'], self.sampler['epi']['range_max'])
        depth_sample = np.around(
            np.random.uniform(self.sampler['depth']['range_min'], self.sampler['depth']['range_max']), decimals=1)

        # Time sampler:
        year = self.par['origin_time'].year  # Constant
        month = self.par['origin_time'].month  # Constant
        day = self.par['origin_time'].day  # Constant
        hour = self.par['origin_time'].hour  # Constant
        sec = int(np.random.uniform(self.sampler['time_range'], self.par['origin_time'].second + 1))
        if sec < int(0):
            sec_new = 59 + sec
            min = self.par['origin_time'].minute - 1  # Constant
        else:
            sec_new = sec
            min = min = self.par['origin_time'].minute
        time_sample = obspy.UTCDateTime(year, month, day, hour, min, sec_new)

        return epi_sample, depth_sample, time_sample

    def model_samples_sdr(self):
        strike_sample = np.random.uniform(self.sampler['strike']['range_min'], self.sampler['strike']['range_max'])
        dip_sample = np.random.uniform(self.sampler['dip']['range_min'], self.sampler['dip']['range_max'])
        rake_sample = np.random.uniform(self.sampler['rake']['range_min'], self.sampler['rake']['range_max'])

        # Time sampler:
        year = self.par['origin_time'].year  # Constant
        month = self.par['origin_time'].month  # Constant
        day = self.par['origin_time'].day  # Constant
        hour = self.par['origin_time'].hour  # Constant
        sec = int(np.random.uniform(self.sampler['time_range'], self.par['origin_time'].second + 1))
        if sec < int(0):
            sec_new = 59 + sec
            min = self.par['origin_time'].minute - 1  # Constant
        else:
            sec_new = sec
            min = min = self.par['origin_time'].minute
        time_sample = obspy.UTCDateTime(year, month, day, hour, min, sec_new)

        return strike_sample, dip_sample, rake_sample, time_sample

    def generate_moment_sdr(self, strike, dip, rake):
        rdip = np.deg2rad(dip)
        rstr = np.deg2rad(strike)
        rrake = np.deg2rad(rake)

        nx = -np.sin(rdip) * np.sin(rstr)
        ny = np.sin(rdip) * np.cos(rstr)
        nz = -np.cos(rdip)

        dx = np.cos(rrake) * np.cos(rstr) + np.cos(rdip) * np.sin(rrake) * np.sin(rstr)
        dy = np.cos(rrake) * np.sin(rstr) - np.cos(rdip) * np.sin(rrake) * np.cos(rstr)
        dz = -np.sin(rdip) * np.sin(rrake)

        Mxx = 2 * dx * nx
        Mxy = dx * ny + dy * nx
        Mxz = dx * nz + dz * nx
        Myy = 2 * dy * ny
        Myz = dy * nz + dz * ny
        Mzz = 2 * dz * nz
        return np.array([Mxx, Myy, Mxy, Mxz, Myz])

    def generate_G(self, epi, depth, t):
        azimuth = self.par['az']

        gf = self.db.get_greens_function(epicentral_distance_in_degree=epi, source_depth_in_m=depth,
                                    origin_time=t, kind=self.par['kind'], kernelwidth=self.par['kernelwidth'],
                                    definition=self.par['definition'])
        tss = gf.traces[0].data
        zss = gf.traces[1].data
        rss = gf.traces[2].data
        tds = gf.traces[3].data
        zds = gf.traces[4].data
        rds = gf.traces[5].data
        zdd = gf.traces[6].data
        rdd = gf.traces[7].data
        zep = gf.traces[8].data
        rep = gf.traces[9].data

        G_z = gf.traces[0].meta['npts']
        G_r = gf.traces[0].meta['npts'] * 2
        G_t = gf.traces[0].meta['npts'] * 3
        G = np.ones((G_t, 5))
        G[0:G_z, 0] = zss * (0.5) * np.cos(2 * np.deg2rad(azimuth)) - zdd * 0.5
        G[0:G_z, 1] = - zdd * 0.5 - zss * (0.5) * np.cos(2 * np.deg2rad(azimuth))
        # G[0:G_z, 1] =  zdd * (1/3) + zep * (1/3)
        G[0:G_z, 2] = zss * np.sin(2 * np.deg2rad(azimuth))
        G[0:G_z, 3] = -zds * np.cos(np.deg2rad(azimuth))
        G[0:G_z, 4] = -zds * np.sin(np.deg2rad(azimuth))

        G[G_z:G_r, 0] = rss * (0.5) * np.cos(2 * np.deg2rad(azimuth)) - rdd * 0.5
        G[G_z:G_r, 1] = -0.5 * rdd - rss * (0.5) * np.cos(2 * np.deg2rad(azimuth))
        # G[G_z:G_r, 1] =  rdd * (1/3) + rep * (1/3)
        G[G_z:G_r, 2] = rss * np.sin(2 * np.deg2rad(azimuth))
        G[G_z:G_r, 3] = -rds * np.cos(np.deg2rad(azimuth))
        G[G_z:G_r, 4] = -rds * np.sin(np.deg2rad(azimuth))

        G[G_r:G_t, 0] = -tss * (0.5) * np.sin(2 * np.deg2rad(azimuth))
        G[G_r:G_t, 1] = tss * (0.5) * np.sin(2 * np.deg2rad(azimuth))
        # G[G_r:G_t, 1] =   0
        G[G_r:G_t, 2] = tss * np.cos(2 * np.deg2rad(azimuth))
        G[G_r:G_t, 3] = tds * np.sin(2 * np.deg2rad(azimuth))
        G[G_r:G_t, 4] = -tds * np.cos(2 * np.deg2rad(azimuth))
        return G

    def G_function(self, epi, depth, t):
        G = self.generate_G(epi, depth, t)
        moment = self.inv.Solve_damping_smoothing(self.d_obs, G)
        # TODO - choose a range for moment with the help of the Resolution Matrix
        d_syn = np.matmul(G, moment)
        return d_syn, moment

    def Window_function(self, epi, depth, t):
        G = self.generate_G(epi, depth, t)
        G_window, d_obs_window, trace_window = self.window.get_windows(self.traces, epi, depth,G)
        moment_window = self.inv.Solve_damping_smoothing(d_obs_window, G_window)
        d_syn_window = np.matmul(G_window, moment_window)
        a = 1
        return d_syn_window, moment_window, d_obs_window, trace_window

    def G_function_sdr(self, strike, dip, rake):
        moment = self.generate_moment_sdr(strike, dip, rake)
        moment.shape = (1, 5)
        self.d_obs.shape = (len(self.d_obs), 1)
        G = np.matmul(self.d_obs, moment)
        # - TODO Solve the maths for the inversion problem
        # G = self.inv.Solve_sdr(self.d_obs,moment)
        d_syn = np.matmul(G, moment.T)
        # plt.plot(d_syn)
        # plt.plot(self.d_obs,linestyle= ':')
        # plt.show()
        return d_syn, moment

    def Window_function_sdr(self, strike, dip, rake):
        moment_window = self.generate_moment_sdr(strike, dip, rake)
        d_obs_window, trace_window = self.window.get_windows(self.traces, self.par['epi'], self.par['depth_s'],
                                                             G_exist=False)
        moment_window.shape = (1, 5)
        d_obs_window.shape = (len(d_obs_window), 1)
        G_window = np.matmul(d_obs_window, moment_window)
        d_syn_window = np.matmul(G_window, moment_window.T)
        return d_syn_window, moment_window, d_obs_window, trace_window

    def write(self, txt_file):
        txt_file.write("%s\n\r" % self.par['VELOC'])  # Velocity model used
        # txt_file.write("%.4f\n\r" % self.par['MO'])  #
        txt_file.write("%.4f\n\r" % self.par['alpha'])  #
        txt_file.write("%.4f\n\r" % self.par['beta'])  #
        txt_file.write("%.4f\n\r" % self.par['az'])  #
        txt_file.write("%i\n\r" % self.par['depth_s'])  #
        txt_file.write("%.4f\n\r" % self.par['epi'])  #
        txt_file.write(
            "%s,%s,%s\n\r" % (self.par['components'][0], self.par['components'][1], self.par['components'][2]))  #
        txt_file.write("%.4f\n\r" % self.par['la_r'])  #
        txt_file.write("%.4f\n\r" % self.par['la_s'])  #
        txt_file.write("%.4f\n\r" % self.par['lo_s'])  #
        txt_file.write("%.4f\n\r" % self.par['lo_r'])  #
        txt_file.write("%.4f\n\r" % self.par['m_pp'])  #
        txt_file.write("%.4f\n\r" % self.par['m_rp'])  #
        txt_file.write("%.4f\n\r" % self.par['m_rr'])  #
        txt_file.write("%.4f\n\r" % self.par['m_rt'])  #
        txt_file.write("%.4f\n\r" % self.par['m_tp'])  #
        txt_file.write("%.4f\n\r" % self.par['m_tt'])  #
        txt_file.write("%i\n\r" % self.par['strike'])  #
        txt_file.write("%i\n\r" % self.par['rake'])  #
        txt_file.write("%i\n\r" % self.par['dip'])  #
        txt_file.write("%s\n\r" % self.par['filter'])  #
        txt_file.write("%s\n\r" % self.par['definition'])  #
        txt_file.write("%s\n\r" % self.par['kind'])  #
        txt_file.write("%s\n\r" % self.par['network'])  #
        txt_file.write("%s\n\r" % self.sampler['filename'])  #
        txt_file.write("%s\n\r" % self.sampler['directory'])  #
        txt_file.write("%s\n\r" % self.sampler['filepath'])  #
        txt_file.write("%i\n\r" % self.sampler['sample_number'])  #
        txt_file.write("%.4f\n\r" % self.sampler['var_est'])  #
        txt_file.write("%i\n\r" % self.sampler['epi']['range_max'])  #
        txt_file.write("%i\n\r" % self.sampler['epi']['range_min'])  #
        txt_file.write("%i\n\r" % self.sampler['epi']['step'])  #
        txt_file.write("%i\n\r" % self.sampler['depth']['range_max'])  #
        txt_file.write("%i\n\r" % self.sampler['depth']['range_min'])  #
        txt_file.write("%i\n\r" % self.sampler['depth']['step'])  #

    def processing(self, savepath, window):
        params = {'legend.fontsize': 'x-large',
                  'figure.figsize': (20, 15),
                  'axes.labelsize': 25,
                  'axes.titlesize': 'x-large',
                  'xtick.labelsize': 25,
                  'ytick.labelsize': 25}
        pylab.rcParams.update(params)
        self.inv = Inversion_problem(self.par)
        if window == True:
            self.window = Source_code(self.par['VELOC_taup'])
        with open(savepath, 'w') as yaml_file:
            self.write(yaml_file)  # Writes all the parameters used for this inversion
            ## Starting parameters and create A START MODEL (MODEL_OLD):
            epi_old, depth_old, time_old = self.model_samples()
            if window == True:
                d_syn_old, moment_old, d_obs, trace_window = self.Window_function(epi_old, depth_old, time_old)

                f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
                trace_z = np.zeros(len(self.traces[0]))
                trace_r = np.zeros(len(self.traces[1]))
                trace_t = np.zeros(len(self.traces[2]))
                trace_z[trace_window['0']['P_min']:trace_window['0']['P_max']] = d_syn_old[0:trace_window['0']['P_len']]
                trace_z[trace_window['0']['S_min']:trace_window['0']['S_max']] = d_syn_old[trace_window['0']['P_len']:
                trace_window['0']['P_len'] + trace_window['0']['S_len']]
                trace_r[trace_window['1']['P_min']:trace_window['1']['P_max']] = d_syn_old[trace_window['0']['P_len'] +
                                                                                           trace_window['0']['S_len']:
                trace_window['0']['P_len'] + trace_window['0']['S_len'] + trace_window['1']['P_len']]
                trace_r[trace_window['1']['S_min']:trace_window['1']['S_max']] = d_syn_old[trace_window['0']['P_len'] +
                                                                                           trace_window['0']['S_len'] +
                                                                                           trace_window['1']['P_len']:
                trace_window['0']['P_len'] + trace_window['0']['S_len'] + trace_window['1']['P_len'] +
                trace_window['1']['S_len']]
                trace_t[trace_window['2']['P_min']:trace_window['2']['P_max']] = d_syn_old[trace_window['0']['P_len'] +
                                                                                           trace_window['0']['S_len'] +
                                                                                           trace_window['1']['P_len'] +
                                                                                           trace_window['1']['S_len']:
                trace_window['0']['P_len'] + trace_window['0']['S_len'] + trace_window['1']['P_len'] +
                trace_window['1']['S_len'] + trace_window['2']['P_len']]
                trace_t[trace_window['2']['S_min']:trace_window['2']['S_max']] = d_syn_old[trace_window['0']['P_len'] +
                                                                                           trace_window['0']['S_len'] +
                                                                                           trace_window['1']['P_len'] +
                                                                                           trace_window['1']['S_len'] +
                                                                                           trace_window['2']['P_len']:]
                trace_z[trace_z == 0] = np.nan
                trace_r[trace_r == 0] = np.nan
                trace_t[trace_t == 0] = np.nan
            else:
                d_obs = self.d_obs
                d_syn_old, moment_old = self.G_function(epi_old, depth_old, time_old)
                trace_z = d_syn_old[0:len(self.traces[0])]
                trace_r = d_syn_old[len(self.traces[0]):len(self.traces[0]) * 2]
                trace_t = d_syn_old[len(self.traces[0]) * 2:len(self.traces[0]) * 3]
                f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
            ax1.plot(trace_z, alpha=0.2)
            ax2.plot(trace_r, alpha=0.2)
            ax3.plot(trace_t, alpha=0.2)

            misfit = Misfit()
            Xi_old = misfit.get_xi(d_obs, d_syn_old, self.var)
            yaml_file.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n\r" % (
                epi_old, depth_old, time_old.timestamp, Xi_old, moment_old[0], moment_old[1],
                moment_old[2], moment_old[3], moment_old[4]))

            for i in range(self.sampler['sample_number']):
                epi, depth, time = self.model_samples()
                if window == True:
                    d_syn, moment, d_obs, trace_window = self.Window_function(epi, depth, time)

                else:
                    d_syn, moment = self.G_function(epi, depth, time)

                misfit = Misfit()
                Xi_new = misfit.get_xi(d_obs, d_syn, self.var)
                random = np.random.random_sample((1,))
                if Xi_new < Xi_old or (Xi_old / Xi_new) > random:
                    if window == True:
                        trace_z = np.zeros(len(self.traces[0]))
                        trace_r = np.zeros(len(self.traces[1]))
                        trace_t = np.zeros(len(self.traces[2]))
                        trace_z[trace_window['0']['P_min']:trace_window['0']['P_max']] = d_syn[
                                                                                         0:trace_window['0']['P_len']]
                        trace_z[trace_window['0']['S_min']:trace_window['0']['S_max']] = d_syn[
                                                                                         trace_window['0']['P_len']:
                                                                                         trace_window['0']['P_len'] +
                                                                                         trace_window['0']['S_len']]
                        trace_r[trace_window['1']['P_min']:trace_window['1']['P_max']] = d_syn[
                                                                                         trace_window['0']['P_len'] +
                                                                                         trace_window['0']['S_len']:
                                                                                         trace_window['0']['P_len'] +
                                                                                         trace_window['0']['S_len'] +
                                                                                         trace_window['1']['P_len']]
                        trace_r[trace_window['1']['S_min']:trace_window['1']['S_max']] = d_syn[
                                                                                         trace_window['0']['P_len'] +
                                                                                         trace_window['0']['S_len'] +
                                                                                         trace_window['1']['P_len']:
                                                                                         trace_window['0']['P_len'] +
                                                                                         trace_window['0']['S_len'] +
                                                                                         trace_window['1']['P_len'] +
                                                                                         trace_window['1']['S_len']]
                        trace_t[trace_window['2']['P_min']:trace_window['2']['P_max']] = d_syn[
                                                                                         trace_window['0']['P_len'] +
                                                                                         trace_window['0']['S_len'] +
                                                                                         trace_window['1']['P_len'] +
                                                                                         trace_window['1']['S_len']:
                                                                                         trace_window['0']['P_len'] +
                                                                                         trace_window['0']['S_len'] +
                                                                                         trace_window['1']['P_len'] +
                                                                                         trace_window['1']['S_len'] +
                                                                                         trace_window['2']['P_len']]
                        trace_t[trace_window['2']['S_min']:trace_window['2']['S_max']] = d_syn[
                                                                                         trace_window['0']['P_len'] +
                                                                                         trace_window['0']['S_len'] +
                                                                                         trace_window['1']['P_len'] +
                                                                                         trace_window['1']['S_len'] +
                                                                                         trace_window['2']['P_len']:]
                        trace_z[trace_z == 0] = np.nan
                        trace_r[trace_r == 0] = np.nan
                        trace_t[trace_t == 0] = np.nan


                    else:
                        d_obs = self.d_obs
                        trace_z = d_syn[0:len(self.traces[0])]
                        trace_r = d_syn[len(self.traces[0]):len(self.traces[0]) * 2]
                        trace_t = d_syn[len(self.traces[0]) * 2:len(self.traces[0]) * 3]

                    ax1.plot(trace_z, alpha=0.2)
                    ax2.plot(trace_r, alpha=0.2)
                    ax3.plot(trace_t, alpha=0.2)
                    yaml_file.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n\r" % (
                        epi, depth, time.timestamp, Xi_new, moment[0], moment[1], moment[2], moment[3],
                        moment[4]))
                    Xi_old = Xi_new
                    epi_old = epi
                    depth_old = depth
                    time_old = time
                    moment_old = moment
                else:
                    yaml_file.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n\r" % (
                        epi_old, depth_old, time_old.timestamp, Xi_old, moment_old[0], moment_old[1],
                        moment_old[2], moment_old[3], moment_old[4]))
                    continue

            self.traces[0].data[self.traces[0].data == 0] = np.nan
            self.traces[1].data[self.traces[1].data == 0] = np.nan
            self.traces[2].data[self.traces[2].data == 0] = np.nan

            ax1.plot(self.traces[0], linestyle=':', label="Observed data")
            ax2.plot(self.traces[1], linestyle=':')
            ax3.plot(self.traces[2], linestyle=':')
            # f.subplots_adjust(hspace=0)
            # plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
            ax1.legend()
            plt.xlabel('Time [s]')
            plt.show()
            plt.savefig(savepath.strip('.txt') + '_%i.pdf' % (self.sampler['sample_number']))
            plt.close()
        yaml_file.close()

    def processing_sdr(self, savepath, window):
        params = {'legend.fontsize': 'x-large',
                  'figure.figsize': (20, 15),
                  'axes.labelsize': 25,
                  'axes.titlesize': 'x-large',
                  'xtick.labelsize': 25,
                  'ytick.labelsize': 25}
        pylab.rcParams.update(params)
        self.inv = Inversion_problem(self.par)
        if window == True:
            self.window = Source_code(self.par['VELOC_taup'])
        with open(savepath, 'w') as yaml_file:
            self.write(yaml_file)  # Writes all the parameters used for this inversion
            ## Starting parameters and create A START MODEL (MODEL_OLD):
            strike_old, dip_old, rake_old, time_old = self.model_samples_sdr()
            if window == True:
                d_syn_old, moment_old, d_obs, trace_window = self.Window_function_sdr(strike_old, dip_old, rake_old)

                f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
                trace_z = np.zeros(len(self.traces[0]))
                trace_r = np.zeros(len(self.traces[1]))
                trace_t = np.zeros(len(self.traces[2]))
                d_syn_old.shape = (len(d_syn_old))
                trace_z[trace_window['0']['P_min']:trace_window['0']['P_max']] = d_syn_old[0:trace_window['0']['P_len']]
                trace_z[trace_window['0']['S_min']:trace_window['0']['S_max']] = d_syn_old[trace_window['0']['P_len']:
                trace_window['0']['P_len'] + trace_window['0']['S_len']]
                trace_r[trace_window['1']['P_min']:trace_window['1']['P_max']] = d_syn_old[trace_window['0']['P_len'] +
                                                                                           trace_window['0']['S_len']:
                trace_window['0']['P_len'] + trace_window['0']['S_len'] + trace_window['1']['P_len']]
                trace_r[trace_window['1']['S_min']:trace_window['1']['S_max']] = d_syn_old[trace_window['0']['P_len'] +
                                                                                           trace_window['0']['S_len'] +
                                                                                           trace_window['1']['P_len']:
                trace_window['0']['P_len'] + trace_window['0']['S_len'] + trace_window['1']['P_len'] +
                trace_window['1']['S_len']]
                trace_t[trace_window['2']['P_min']:trace_window['2']['P_max']] = d_syn_old[trace_window['0']['P_len'] +
                                                                                           trace_window['0']['S_len'] +
                                                                                           trace_window['1']['P_len'] +
                                                                                           trace_window['1']['S_len']:
                trace_window['0']['P_len'] + trace_window['0']['S_len'] + trace_window['1']['P_len'] +
                trace_window['1']['S_len'] + trace_window['2']['P_len']]
                trace_t[trace_window['2']['S_min']:trace_window['2']['S_max']] = d_syn_old[trace_window['0']['P_len'] +
                                                                                           trace_window['0']['S_len'] +
                                                                                           trace_window['1']['P_len'] +
                                                                                           trace_window['1']['S_len'] +
                                                                                           trace_window['2']['P_len']:]
                trace_z[trace_z == 0] = np.nan
                trace_r[trace_r == 0] = np.nan
                trace_t[trace_t == 0] = np.nan
                d_obs.shape = (len(d_obs))
            else:
                d_obs = self.d_obs
                d_syn_old, moment_old = self.G_function_sdr(strike_old, dip_old, rake_old)
                trace_z = d_syn_old[0:len(self.traces[0])]
                trace_r = d_syn_old[len(self.traces[0]):len(self.traces[0]) * 2]
                trace_t = d_syn_old[len(self.traces[0]) * 2:len(self.traces[0]) * 3]
                f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
            # plt.plot(d_syn_old, alpha=0.2)
            ax1.plot(trace_z, alpha=0.2)
            ax2.plot(trace_r, alpha=0.2)
            ax3.plot(trace_t, alpha=0.2)

            misfit = Misfit()
            Xi_old = misfit.get_xi(d_obs, d_syn_old, self.sampler['var_est'])

            yaml_file.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n\r" % (
                strike_old, dip_old, rake_old, time_old.timestamp, Xi_old, moment_old[0][0], moment_old[0][1],
                moment_old[0][2], moment_old[0][3], moment_old[0][4]))

            for i in range(self.sampler['sample_number']):
                strike, dip, rake, time = self.model_samples_sdr()
                if window == True:
                    d_syn, moment, d_obs, trace_window = self.Window_function_sdr(strike, dip, rake)

                else:
                    self.d_obs = d_obs
                    d_syn, moment = self.G_function_sdr(strike, dip, rake)

                misfit = Misfit()
                Xi_new = misfit.get_xi(d_obs, d_syn, self.sampler['var_est'])
                random = np.random.random_sample((1,))
                if Xi_new < Xi_old or (Xi_old / Xi_new) > random:
                    if window == True:
                        trace_z = np.zeros(len(self.traces[0]))
                        trace_r = np.zeros(len(self.traces[1]))
                        trace_t = np.zeros(len(self.traces[2]))
                        d_syn.shape = (len(d_syn_old))
                        trace_z[trace_window['0']['P_min']:trace_window['0']['P_max']] = d_syn[
                                                                                         0:trace_window['0']['P_len']]
                        trace_z[trace_window['0']['S_min']:trace_window['0']['S_max']] = d_syn[
                                                                                         trace_window['0']['P_len']:
                                                                                         trace_window['0']['P_len'] +
                                                                                         trace_window['0']['S_len']]
                        trace_r[trace_window['1']['P_min']:trace_window['1']['P_max']] = d_syn[
                                                                                         trace_window['0']['P_len'] +
                                                                                         trace_window['0']['S_len']:
                                                                                         trace_window['0']['P_len'] +
                                                                                         trace_window['0']['S_len'] +
                                                                                         trace_window['1']['P_len']]
                        trace_r[trace_window['1']['S_min']:trace_window['1']['S_max']] = d_syn[
                                                                                         trace_window['0']['P_len'] +
                                                                                         trace_window['0']['S_len'] +
                                                                                         trace_window['1']['P_len']:
                                                                                         trace_window['0']['P_len'] +
                                                                                         trace_window['0']['S_len'] +
                                                                                         trace_window['1']['P_len'] +
                                                                                         trace_window['1']['S_len']]
                        trace_t[trace_window['2']['P_min']:trace_window['2']['P_max']] = d_syn[
                                                                                         trace_window['0']['P_len'] +
                                                                                         trace_window['0']['S_len'] +
                                                                                         trace_window['1']['P_len'] +
                                                                                         trace_window['1']['S_len']:
                                                                                         trace_window['0']['P_len'] +
                                                                                         trace_window['0']['S_len'] +
                                                                                         trace_window['1']['P_len'] +
                                                                                         trace_window['1']['S_len'] +
                                                                                         trace_window['2']['P_len']]
                        trace_t[trace_window['2']['S_min']:trace_window['2']['S_max']] = d_syn[
                                                                                         trace_window['0']['P_len'] +
                                                                                         trace_window['0']['S_len'] +
                                                                                         trace_window['1']['P_len'] +
                                                                                         trace_window['1']['S_len'] +
                                                                                         trace_window['2']['P_len']:]
                        trace_z[trace_z == 0] = np.nan
                        trace_r[trace_r == 0] = np.nan
                        trace_t[trace_t == 0] = np.nan
                        d_obs.shape = (len(d_obs))


                    else:
                        d_obs = self.d_obs
                        trace_z = d_syn[0:len(self.traces[0])]
                        trace_r = d_syn[len(self.traces[0]):len(self.traces[0]) * 2]
                        trace_t = d_syn[len(self.traces[0]) * 2:len(self.traces[0]) * 3]
                        trace_z[trace_z == 0] = np.nan
                        trace_r[trace_r == 0] = np.nan
                        trace_t[trace_t == 0] = np.nan
                    # plt.plot(d_syn, alpha=0.2)
                    ax1.plot(trace_z, alpha=0.2)
                    ax2.plot(trace_r, alpha=0.2)
                    ax3.plot(trace_t, alpha=0.2)
                    yaml_file.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n\r" % (
                        strike_old, dip_old, rake_old, time_old.timestamp, Xi_old, moment_old[0][0], moment_old[0][1],
                        moment_old[0][2], moment_old[0][3], moment_old[0][4]))
                    Xi_old = Xi_new
                    strike_old = strike
                    dip_old = dip
                    rake_old = rake
                    moment_old = moment
                    time_old = time

                else:
                    yaml_file.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n\r" % (
                        strike_old, dip_old, rake_old, time_old.timestamp, Xi_old, moment_old[0][0], moment_old[0][1],
                        moment_old[0][2], moment_old[0][3], moment_old[0][4]))
                    # plt.plot(d_syn, alpha=0.2)
                    continue

            self.traces[0].data[self.traces[0].data == 0] = np.nan
            self.traces[1].data[self.traces[1].data == 0] = np.nan
            self.traces[2].data[self.traces[2].data == 0] = np.nan

            ax1.plot(self.traces[0], linestyle=':', label="Observed data")
            ax2.plot(self.traces[1], linestyle=':')
            ax3.plot(self.traces[2], linestyle=':')
            ax1.legend()
            # plt.plot(self.d_obs, ":")
            plt.xlabel('Time [s]')
            plt.show()
            plt.savefig(savepath.strip('.txt') + '_%i.pdf' % (self.sampler['sample_number']))
            plt.close()
        yaml_file.close()

    def test_samples(self):
        # This function just saves the different samples, important to check how we sample!
        self.inv = Inversion_problem(self.par)
        filepath = self.sampler['directory'] + '/sampler_test.txt'
        with open(filepath, 'w') as yaml_file:
            ## Starting parameters and create A START MODEL (MODEL_OLD):
            epi_old, depth_old, time_old = self.model_samples()
            yaml_file.write("%.4f,%.4f,%.4f\n\r" % (epi_old, depth_old, time_old.timestamp))

            for i in range(1000):
                epi, depth, time = self.model_samples()
                yaml_file.write("%.4f,%.4f,%.4f\n\r" % (epi, depth, time.timestamp))
        yaml_file.close()

    # ---------------------------------------------------------------------------------------------------------------------#
    #                                                 RUN parallel                                                         #
    # ---------------------------------------------------------------------------------------------------------------------#
    def do_parallel(self, window=True, sdr=False):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        print("Rank", rank, "Size", size)
        if sdr == True:
            dir_proc = self.sampler['directory'] + '/proc_sds'
        else:
            dir_proc = self.sampler['directory'] + '/proc'
        if not os.path.exists(dir_proc):
            os.makedirs(dir_proc)
        filepath_proc = dir_proc + '/file_proc_%i.txt' % rank
        if sdr == True:
            try:
                self.processing_sdr(filepath_proc, window=window)
            except TypeError:
                print ("TypeError in rank: %i" % rank)
                self.processing_sdr(filepath_proc, window=window)
        else:
            try:
                self.processing(filepath_proc, window=window)
            except TypeError:
                print ("TypeError in rank: %i" % rank)
                self.processing(filepath_proc, window=window)

        comm.bcast()  # All the processor wait until they are all at this point
        if rank == 0:
            print ("The .txt files are saved in: %s" % dir_proc)