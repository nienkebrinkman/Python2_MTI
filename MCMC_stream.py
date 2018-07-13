import mpi4py as MPI
from mpi4py import MPI
import os
import numpy as np
import geographiclib.geodesic as geo
import seaborn as sns
import matplotlib.pylab as plt
import instaseis
from obspy.core.stream import Stream
import obspy

from Source_code import Source_code
from Seismogram import Seismogram
from Misfit import Misfit
from Surface_waves import Surface_waves
from Blindtest import Blindtest


class MCMC_stream:
    def __init__(self, R_env_obs, L_env_obs, total_traces_obs, P_traces_obs, S_traces_obs, PRIOR, db,
                 specification_values, time_at_receiver, start_sample_path=None, picked_events=None,
                 full_obs_trace=None, P_start=None, S_start=None):
        # specification values is a dict of the following, See Get_Parameters.py to see the form of values:
        self.blind = specification_values['blind']
        self.sdr = specification_values['sdr']
        self.plot_modus = specification_values['plot_modus']
        self.xi = specification_values['misfit']
        self.MCMC = specification_values['MCMC']
        self.temperature = specification_values['temperature']
        self.directory = specification_values['directory']
        self.noise = specification_values['noise']
        self.npts = specification_values['npts']

        self.R_env_obs = R_env_obs
        self.L_env_obs = L_env_obs
        self.total_tr_obs = total_traces_obs
        self.s_tr_obs = S_traces_obs
        self.p_tr_obs = P_traces_obs
        self.prior = PRIOR
        self.db = db
        if full_obs_trace == None:
            pass
        else:
            self.full_obs_trace = full_obs_trace
            self.S_start = S_start
            self.P_Start = P_start

        # General parameters:
        self.dt = total_traces_obs[0].meta.delta
        self.temperature = specification_values['temperature']
        self.time_at_rec = time_at_receiver
        self.start_sample = start_sample_path
        self.rnd_par = specification_values['rnd_par']

        # Initiate all the python codes needed:
        self.window = Source_code(self.prior['VELOC_taup'])
        self.seis = Seismogram(self.prior, self.db)
        self.misfit = Misfit(specification_values['directory'])
        if self.blind == True:
            self.BT = Blindtest()
            self.picks = picked_events
        else:
            self.SW = Surface_waves(self.prior)

    def model_samples(self, epi_old=None, depth_old=None):
        if epi_old == None or depth_old == None:
            epi_sample = np.random.uniform(self.prior['epi']['range_min'], self.prior['epi']['range_max'])
            depth_sample = np.around(
                np.random.uniform(self.prior['depth']['range_min'], self.prior['depth']['range_max']), decimals=1)
        else:
            if self.update == 'epi':
                epi_sample = np.random.normal(epi_old, self.prior['epi']['spread'])
                depth_sample = depth_old
            elif self.update == 'depth':
                epi_sample = epi_old
                depth_sample = np.around(np.random.normal(depth_old, self.prior['depth']['spread']), decimals=1)
            else:
                epi_sample = epi_old
                depth_sample = depth_old
        return epi_sample, depth_sample

    def model_samples_sdr(self, strike_old=None, dip_old=None, rake_old=None):
        if strike_old == None or dip_old == None or rake_old == None:
            strike = np.random.uniform(self.prior['strike']['range_min'], self.prior['strike']['range_max'])
            dip = np.random.uniform(self.prior['dip']['range_min'], self.prior['dip']['range_max'])
            rake = np.random.uniform(self.prior['rake']['range_min'], self.prior['rake']['range_max'])

        else:
            if self.update == 'moment':
                # Change radian to degree
                dip_rad = np.deg2rad(dip_old)
                strike_rad = np.deg2rad(strike_old)

                # Calculate normal vector of Fault geometry using strike and dip:
                n = np.array(
                    [-np.sin(dip_rad) * np.sin(strike_rad), -np.sin(dip_rad) * np.cos(strike_rad), np.cos(dip_rad)])

                # X,Y,Z coordinate of the Northpole:
                north_coor = np.array([0, 0, 1])

                # Rotation Axis of from Northpole to Old_sample
                R = np.cross(north_coor, n)
                R_norm = R / (np.sqrt(np.sum(np.square(R), axis=0)))

                # New proposal angle which depends on a spread specified:
                random_phi = np.abs(np.random.normal(0, self.prior['angle_spread']))
                phi = np.deg2rad(random_phi)

                # Theta will be choosen from a point on a circle all with epicentral radius of: Phi
                random_theta = np.random.choice(
                    np.around(np.linspace(0, 360, 361),
                              decimals=1))
                theta = np.deg2rad(random_theta)

                # X,Y,Z coordinates of the new_sample, BUT looking from the northpole (SO STILL NEEDS ROTATION)
                new_coor = np.array([-np.sin(phi) * np.sin(theta), -np.sin(phi) * np.cos(theta), np.cos(phi)])

                # X,Y,Z coordinates with rotation included --> using: Rodrigues' Rotation Formula
                beta = dip_rad
                R_new_coor = np.cos(beta) * new_coor + np.sin(beta) * (np.cross(R_norm, new_coor)) + (np.inner(R_norm,
                                                                                                               new_coor)) * (
                                                                                                         1 - np.cos(
                                                                                                             beta)) * R_norm

                # Determine from a normal distribution a new rake: mean = old rake and SD = spread
                rake = np.random.normal(rake_old, self.prior['rake']['spread'])

                if rake < -180:
                    rake = (180 + rake) % 180
                if rake == 180:
                    rake = -rake
                if rake > 179:
                    rake = (180 + rake) % -180

                phi_normal = np.rad2deg(np.arctan2(R_new_coor[1], R_new_coor[0]))
                wrap = phi_normal % 360
                st = 360.0 - 90.0 - wrap
                strike = st % 360.0

                dip = np.rad2deg(np.arctan2(np.sqrt(R_new_coor[0] ** 2 + R_new_coor[1] ** 2), R_new_coor[2]))
                # dip = 90.0
                if dip >= 90 or dip < 0:
                    if dip == 90:
                        dip = 89.0
                        strike = (strike + 180.0) % 360.0
                        if rake == -180:
                            pass
                        else:
                            rake = -rake
                    else:
                        d_new = 90.0 - dip
                        dip = d_new % 90.0
                        strike = (strike + 180.0) % 360.0
                        if rake == -180:
                            pass
                        else:
                            rake = -rake
            else:
                strike = strike_old
                dip = dip_old
                rake = rake_old
        return strike, dip, rake

    def G_function(self, epi, depth, moment_old=None):
        dict = geo.Geodesic(a=self.prior['radius'], f=0).ArcDirect(lat1=self.prior['la_r'], lon1=self.prior['lo_r'],
                                                                   azi1=self.prior['baz'],
                                                                   a12=epi, outmask=1929)
        if moment_old is None:
            strike, dip, rake = self.model_samples_sdr()
        else:
            strike, dip, rake = self.model_samples_sdr(moment_old[0], moment_old[1], moment_old[2])
        d_syn, traces_syn, sources = self.seis.get_seis_manual(la_s=dict['lat2'], lo_s=dict['lon2'], depth=depth,
                                                               strike=strike, dip=dip, rake=rake,
                                                               time=self.time_at_rec, sdr=self.sdr)
        if self.blind == True:
            R_env_syn, L_env_syn = self.BT.pick_sw(traces_syn, self.picks, epi, self.prior, 30000, self.directory,
                                                   plot_modus=False)
        else:
            R_env_syn = self.SW.rayleigh_pick(Z_trace=traces_syn.traces[0], la_s=dict['lat2'], lo_s=dict['lon2'],
                                              depth=depth, save_directory=self.directory, time_at_rec=self.time_at_rec,
                                              npts=self.npts, plot_modus=False)
            L_env_syn = self.SW.love_pick(T_trace=traces_syn.traces[2], la_s=dict['lat2'], lo_s=dict['lon2'],
                                          depth=depth, save_directory=self.directory, time_at_rec=self.time_at_rec,
                                          npts=self.npts, plot_modus=False)
        traces_syn.plot(outfile=self.directory + '/syntethic')
        total_syn, p_syn, s_syn, start_time_p, start_time_s = self.window.get_window_obspy(traces_syn, epi, depth,
                                                                                           self.time_at_rec, self.npts)

        plot_obs = self.full_obs_trace.copy()
        plot_obs.trim(traces_syn.traces[0].meta.starttime, traces_syn.traces[0].meta.endtime)
        if self.iter % 10 == 0:
            for i in range(len(self.full_obs_trace)):
                subplot_no = len(self.full_obs_trace) * 100 + 10 + i + 1
                if i < 3:
                    P_obs = int(
                        (self.P_Start.timestamp - traces_syn.traces[
                            i].meta.starttime.timestamp) / traces_syn.traces[i].meta.delta)
                    P_syn = int(
                        (start_time_p.timestamp - traces_syn.traces[
                            i].meta.starttime.timestamp) / traces_syn.traces[i].meta.delta)
                S_obs = int(
                    (self.S_start.timestamp - traces_syn.traces[
                        i].meta.starttime.timestamp) / traces_syn.traces[i].meta.delta)

                S_syn = int(
                    (start_time_s.timestamp - traces_syn.traces[
                        i].meta.starttime.timestamp) / traces_syn.traces[i].meta.delta)
                ax = plt.subplot(subplot_no)
                plt.plot(plot_obs.traces[i].data, alpha=0.5, c='k', linewidth=0.3)
                plt.plot(traces_syn.traces[i].data, alpha=0.5, c='r', linewidth=0.3)
                ymin, ymax = ax.get_ylim()
                if i < 3:
                    plt.vlines([P_obs], ymin, ymax, label="Observed", colors='k', linewidth=0.3)
                    plt.vlines([P_syn], ymin, ymax, label="synthetic", colors='r', linewidth=0.3)
                plt.vlines([S_obs], ymin, ymax, label="Observed", colors='k', linewidth=0.3)
                plt.vlines([S_syn], ymin, ymax, label="synthetic", colors='r', linewidth=0.3)
                plt.xlabel(self.time_at_rec.strftime('%Y-%m-%dT%H:%M:%S + sec'))
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.tight_layout()

            plt.savefig(self.directory + '/P_S_Picks_%i.pdf' %i)
            # plt.show()
            plt.close()

        return R_env_syn, L_env_syn, total_syn, p_syn, s_syn, np.array([strike, dip, rake])

    def start_MCMC(self, savepath):
        savepath_reject = savepath.replace('.txt', '_reject.txt')
        savepath_shift = savepath.replace('.txt', '_shift.txt')
        with open(savepath_shift, 'w') as shift_file:
            with open(savepath, 'w') as save_file:
                with open(savepath_reject, 'w') as save_reject_file:
                    self.write_par(save_file)
                    accepted = 0
                    rejected = 0
                    for i in range(self.prior['sample_number']):
                        self.iter = i
                        if i % 10 == 0:
                            print("proposal: %i, accepted: %i" % (i, accepted))
                            print("proposal: %i, rejected: %i" % (i, rejected))

                        if i == 0 or self.MCMC == 'MH':
                            if self.start_sample == None:
                                epi, depth = self.model_samples()
                                R_env_syn, L_env_syn, total_syn, p_syn, s_syn, moment = self.G_function(epi, depth)
                            else:
                                if self.rnd_par == True:
                                    self.update = 'epi'
                                else:
                                    self.update = None
                                data = np.loadtxt(self.start_sample, delimiter=',')
                                epi = data[0]
                                depth = data[1]
                                moment_old = np.array([data[2], data[3], data[4]])
                                R_env_syn, L_env_syn, total_syn, p_syn, s_syn, moment = self.G_function(epi, depth,
                                                                                                        moment_old)

                        else:
                            if self.rnd_par == True:
                                self.update = np.random.choice(['epi', 'depth', 'moment'], 1)[0]
                            else:
                                self.update = None
                            epi, depth = self.model_samples(epi_old, depth_old)
                            if epi < self.prior['epi']['range_min'] or epi > self.prior['epi']['range_max'] or depth < \
                                    self.prior['depth']['range_min'] or depth > self.prior['depth']['range_max']:
                                continue
                            R_env_syn, L_env_syn, total_syn, p_syn, s_syn, moment = self.G_function(epi, depth,
                                                                                                    moment_old)

                        if self.xi == 'L2':
                            Xi_bw, time_shift_new = self.misfit.L2_stream(self.p_tr_obs, p_syn, self.s_tr_obs, s_syn,
                                                                          self.time_at_rec, self.prior['var_est'])
                            Xi_R = self.misfit.SW_L2(self.R_env_obs, R_env_syn, self.prior['var_est'])
                            Xi_L = self.misfit.SW_L2(self.L_env_obs, L_env_syn, self.prior['var_est'])
                            Xi_new = Xi_bw + Xi_R + Xi_L
                        elif self.xi == 'CC':
                            Xi_bw_new, time_shift_new = self.misfit.CC_stream(self.p_tr_obs, p_syn, self.s_tr_obs,
                                                                              s_syn,
                                                                              self.time_at_rec)
                            shift_file.write("%.4f,%.4f,%.4f,%.4f,%.4f\n" % (
                            time_shift_new[0], time_shift_new[1], time_shift_new[2], time_shift_new[3],
                            time_shift_new[4]))
                            s_z_new = 0.1 * Xi_bw_new[0]
                            s_r_new = 0.1 * Xi_bw_new[1]
                            s_t_new = 5 * Xi_bw_new[2]
                            p_z_new = 5 * Xi_bw_new[3]
                            p_r_new = 5 * Xi_bw_new[4]
                            bw_new = s_z_new + s_r_new + s_t_new + p_z_new + p_r_new
                            Xi_R_new = self.misfit.SW_L2(self.R_env_obs, R_env_syn, self.prior['var_est'])
                            R_dict_new = {}
                            rw_new = 0
                            for j, v in enumerate(Xi_R_new):
                                R_dict_new.update({'R_%i_new' % j: 0.1*v})
                                rw_new += v

                            Xi_L_new = self.misfit.SW_L2(self.L_env_obs, L_env_syn, self.prior['var_est'])
                            L_dict_new = {}
                            lw_new = 0
                            for j, v in enumerate(Xi_L_new):
                                L_dict_new.update({'L_%i_new' % j: 0.1*v})
                                lw_new += v
                            Xi_new = bw_new + rw_new + lw_new
                        else:
                            raise ValueError(
                                'misfit is not specified correctly, choose either: L2 or CC in string format')

                        if i == 0:
                            s_z_old = s_z_new
                            s_r_old = s_r_new
                            s_t_old = s_t_new
                            p_z_old = p_z_new
                            p_r_old = p_r_new
                            bw_old = bw_new
                            R_dict_old = {}
                            for j, v in R_dict_new.iteritems():
                                R_dict_old.update({j.replace("new", "old"): v})
                            rw_old = rw_new
                            L_dict_old = {}
                            for j, v in L_dict_new.iteritems():
                                L_dict_old.update({j.replace("new", "old"): v})
                            lw_old = lw_new
                            Xi_old = Xi_new

                            epi_old = epi
                            depth_old = depth
                            moment_old = moment
                            self.write_sample(save_file, epi_old, depth_old, moment_old, Xi_old, s_z_old, s_r_old,
                                              s_t_old, p_z_old, p_r_old, bw_old, R_dict_old, rw_old, L_dict_old, lw_old,
                                              accept=1)
                            self.write_sample(save_reject_file, epi_old, depth_old, moment_old, Xi_old, s_z_old,
                                              s_r_old, s_t_old, p_z_old, p_r_old, bw_old, R_dict_old, rw_old,
                                              L_dict_old, lw_old, accept=1)

                            if self.plot_modus == True:
                                plot_traces = p_syn.__add__(s_syn)
                                plot_bw = Stream()
                                plot_bw_reject = Stream()
                                plot_R = Stream()
                                plot_R_reject = Stream()
                                plot_L = Stream()
                                plot_L_reject = Stream()

                                plot_bw += plot_traces
                                plot_bw_reject += plot_traces
                                plot_R += R_env_syn
                                plot_R_reject += R_env_syn
                                plot_L_reject += L_env_syn
                                plot_L += L_env_syn

                                self.total_tr_obs.write(self.directory + '/bw.mseed', format='MSEED')
                                self.R_env_obs.write(self.directory + '/R.mseed', format='MSEED')
                                self.L_env_obs.write(self.directory + '/L.mseed', format='MSEED')
                                self.total_tr_obs.write(self.directory + '/bw_reject.mseed', format='MSEED')
                                self.R_env_obs.write(self.directory + '/R_reject.mseed', format='MSEED')
                                self.L_env_obs.write(self.directory + '/L_reject.mseed', format='MSEED')

                                plot_bw.write(self.directory + '/bw.mseed', format='MSEED')
                                plot_R.write(self.directory + '/R.mseed', format='MSEED')
                                plot_L.write(self.directory + '/L.mseed', format='MSEED')
                                plot_bw_reject.write(self.directory + '/bw_reject.mseed', format='MSEED')
                                plot_R_reject.write(self.directory + '/R_reject.mseed', format='MSEED')
                                plot_L_reject.write(self.directory + '/L_reject.mseed', format='MSEED')

                            continue

                        random = np.random.random_sample((1,))
                        if (Xi_new < Xi_old) or (np.exp((Xi_old - Xi_new) / self.temperature) > random):
                            s_z_old = s_z_new
                            s_r_old = s_r_new
                            s_t_old = s_t_new
                            p_z_old = p_z_new
                            p_r_old = p_r_new
                            bw_old = bw_new
                            R_dict_old = {}
                            for j, v in R_dict_new.iteritems():
                                R_dict_old.update({j.replace("new", "old"): v})
                            rw_old = rw_new
                            L_dict_old = {}
                            for j, v in L_dict_new.iteritems():
                                L_dict_old.update({j.replace("new", "old"): v})
                            lw_old = lw_new
                            Xi_old = Xi_new

                            epi_old = epi
                            depth_old = depth
                            moment_old = moment
                            accepted = accepted + 1
                            self.write_sample(save_file, epi_old, depth_old, moment_old, Xi_old, s_z_old, s_r_old,
                                              s_t_old, p_z_old, p_r_old, bw_old, R_dict_old, rw_old, L_dict_old, lw_old,
                                              accept=1)
                            if i % 10 == 0:

                                if self.plot_modus == True:
                                    plot_traces = plot_traces.copy()
                                    plot_traces = p_syn.__add__(s_syn)
                                    plot_bw += plot_traces
                                    plot_R += R_env_syn
                                    plot_L += L_env_syn
                                    plot_bw.write(self.directory + '/bw.mseed', format='MSEED')
                                    plot_R.write(self.directory + '/R.mseed', format='MSEED')
                                    plot_L.write(self.directory + '/L.mseed', format='MSEED')


                        else:
                            rejected += 1
                            self.write_sample(save_file, epi, depth, moment, Xi_old, s_z_old, s_r_old, s_t_old, p_z_old,
                                              p_r_old, bw_old, R_dict_old, rw_old, L_dict_old, lw_old, accept=0)
                            self.write_sample(save_reject_file, epi, depth, moment, Xi_new, s_z_new, s_r_new, s_t_new,
                                              p_z_new, p_r_new, bw_new, R_dict_new, rw_new, L_dict_new, lw_new,
                                              accept=0)
                            if i % 10 == 0:
                                if self.plot_modus == True:
                                    plot_traces = plot_traces.copy()
                                    plot_traces = p_syn.__add__(s_syn)
                                    plot_bw_reject += plot_traces
                                    plot_R_reject += R_env_syn
                                    plot_L_reject += L_env_syn
                                    plot_bw_reject.write(self.directory + '/bw_reject.mseed', format='MSEED')
                                    plot_R_reject.write(self.directory + '/R_reject.mseed', format='MSEED')
                                    plot_L_reject.write(self.directory + '/L_reject.mseed', format='MSEED')

                    if self.plot_modus == True:
                        plot_obs_bw = self.p_tr_obs.__add__(self.s_tr_obs)
                        plot_obs_R = self.R_env_obs
                        plot_obs_L = self.L_env_obs
                        self.plot_streams(plot_obs_bw, plot_bw, 1)
                        self.plot_streams(plot_obs_R, plot_R, 2)
                        self.plot_streams(plot_obs_L, plot_L, 3)
                        self.plot_streams(plot_obs_bw, plot_bw_reject, 4)
                        self.plot_streams(plot_obs_R, plot_R_reject, 5)
                        self.plot_streams(plot_obs_L, plot_L_reject, 6)

                        # self.plot(1, plot_obs, label="bw", iter="final", accepted=True)
                        # self.plot(2, self.R_env_obs, label="R", iter="final", accepted=True)
                        # self.plot(3, self.L_env_obs, label="L", iter="final", accepted=True)
                        #
                        # self.plot(4, plot_obs, label="bw", iter="final", accepted=False)
                        # self.plot(5, self.R_env_obs, label="R", iter="final", accepted=False)
                        # self.plot(6, self.L_env_obs, label="L", iter="final", accepted=False)

                save_file.close()
            save_reject_file.close()
        shift_file.close

    def plot_streams(self, obs_stream, syn_stream, fig_num):
        label, save_name = self.get_label(fig_num, len(obs_stream))
        plt.figure(fig_num)
        length = len(syn_stream) / len(obs_stream)
        for k in range(length):
            iter = 0
            for j in range(len(obs_stream)):
                p = j + k * len(obs_stream)
                subplot_no = len(obs_stream) * 100 + 10 + iter + 1
                plt.subplot(subplot_no)
                if k == 0:
                    plt.plot(self.zero_to_nan(syn_stream.traces[p].data), linewidth=0.3,
                             label="start_sample:%s" % label[j], linestyle=":", c='r')
                elif k == length - 1:
                    plt.plot(self.zero_to_nan(obs_stream.traces[j].data), alpha=0.5, linewidth=0.3,
                             label="obs_sample:%s" % label[j], c='k')
                    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
                    plt.tight_layout()
                else:
                    plt.plot(self.zero_to_nan(syn_stream.traces[p].data),  linewidth=0.3, c='k')
                iter += 1
        plt.savefig(self.directory + save_name)
        plt.close()

    def get_label(self, fig_num, len_obs_stream):
        if fig_num == 1:
            label = np.array(["P_z", "P_r", "S_z", "S_r", "S_t"])
            save_name = '/Accepted_bw.pdf'
        elif fig_num == 2:
            label = np.array([])
            for i in range(len_obs_stream):
                label = np.append(label, "R_%i" % i)
            save_name = '/Accepted_R.pdf'
        elif fig_num == 3:
            label = np.array([])
            for i in range(len_obs_stream):
                label = np.append(label, "L_%i" % i)
            save_name = '/Accepted_L.pdf'
        elif fig_num == 4:
            label = np.array(["P_z", "P_r", "S_z", "S_r", "S_t"])
            save_name = '/Rejected_bw.pdf'
        elif fig_num == 5:
            label = np.array([])
            for i in range(len_obs_stream):
                label = np.append(label, "R_%i" % i)
            save_name = '/Rejected_R.pdf'
        elif fig_num == 6:
            label = np.array([])
            for i in range(len_obs_stream):
                label = np.append(label, "L_%i" % i)
            save_name = '/Rejected_L.pdf'
        return label, save_name

    def plot(self, figure_number, stream_to_plot, label, iter, accepted=True):
        # -- Label -- [String]
        # bw - P_z,P_r,S_z,S_r,S_t
        # R  - 10_20,08-16,16-32,24-48
        # L  - 24_48,16_32,12_24,08_16
        # None- No label

        # -- Iter -- [String]
        # zero - First iteration --> staring sample in label
        # final- Last iteration --> observed sample in label and saves/closes the figure
        # None - Nothing happens
        if label == "bw":
            lab = np.array(["P_z", "P_r", "S_z", "S_r", "S_t"])
        elif label == "R":
            lab = np.array(["10_20", "08-16", "16-32", "24-48"])
        elif label == "L":
            lab = np.array(["24_48", "16_32", "12_24", "08_16"])
        else:
            lab = None

        plt.figure(figure_number)
        for i in range(len(stream_to_plot)):
            subplot_no = len(stream_to_plot) * 100 + 10 + i + 1
            plt.subplot(subplot_no)
            if lab is None:
                plt.plot(self.zero_to_nan(stream_to_plot.traces[i].data), alpha=0.5, linewidth=0.3, c='b')
            else:
                if iter == "zero":
                    plt.plot(self.zero_to_nan(stream_to_plot.traces[i].data), alpha=0.5, linewidth=0.3,
                             label="start_sample:%s" % lab[i], linestyle=":", c='r')
                elif iter == "final":
                    plt.plot(self.zero_to_nan(stream_to_plot.traces[i].data), alpha=0.5, linewidth=0.3,
                             label="obs_sample:%s" % lab[i], c='k')
                    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
                    plt.tight_layout()
        if iter == "final":
            if accepted == True:
                plt.savefig(self.directory + '/Accepted_%s.pdf' % label)
            else:
                plt.savefig(self.directory + '/Rejected_%s.pdf' % label)

    def zero_to_nan(self, values):
        """Replace every 0 with 'nan' and return a copy."""
        return [float('nan') if x == 0 else x for x in values]

    def write_par(self, txt_file):
        txt_file.write("Velocity model:%s\n\r" % self.prior['VELOC'])  # Velocity model used
        txt_file.write("Inversion method:%s\n\r" % self.MCMC)
        txt_file.write("Misfit used for BW:%s\n\r" % self.xi)
        txt_file.write("noise:%r\n\r" % self.noise)
        txt_file.write("sdr:%r\n\r" % self.sdr)
        txt_file.write("plot_modus:%r\n\r" % self.plot_modus)
        # txt_file.write("%.4f\n\r" % self.par['MO'])  #
        txt_file.write("alpha:%.4f\n\r" % self.prior['alpha'])  #
        txt_file.write("beta:%.4f\n\r" % self.prior['beta'])  #
        txt_file.write("azimuth:%.4f\n\r" % self.prior['az'])  #
        txt_file.write(
            "%s,%s,%s\n\r" % (self.prior['components'][0], self.prior['components'][1], self.prior['components'][2]))  #
        txt_file.write("la_r:%.4f\n\r" % self.prior['la_r'])  #
        txt_file.write("lo_r:%.4f\n\r" % self.prior['lo_r'])  #
        txt_file.write("filter:%s\n\r" % self.prior['filter'])  #
        txt_file.write("definition:%s\n\r" % self.prior['definition'])  #
        txt_file.write("kind:%s\n\r" % self.prior['kind'])  #
        txt_file.write("network:%s\n\r" % self.prior['network'])  #
        txt_file.write("amount samples:%i\n\r" % self.prior['sample_number'])  #
        txt_file.write("Temperature:%i\n\r" % self.temperature)  #
        txt_file.write("variance est:%.4f\n\r" % self.prior['var_est'])  #
        txt_file.write("epi_range_max:%i\n\r" % self.prior['epi']['range_max'])  #
        txt_file.write("epi_range_min:%i\n\r" % self.prior['epi']['range_min'])  #
        txt_file.write("epi_spread:%i\n\r" % self.prior['epi']['spread'])  #
        txt_file.write("depth_range_max:%i\n\r" % self.prior['depth']['range_max'])  #
        txt_file.write("depth_range_min:%i\n\r" % self.prior['depth']['range_min'])  #
        txt_file.write("depth_spread:%i\n\r" % self.prior['depth']['spread'])
        txt_file.write("strike_range_max:%i\n\r" % self.prior['strike']['range_max'])
        txt_file.write("strike_range_min:%i\n\r" % self.prior['strike']['range_min'])
        txt_file.write("dip_range_max:%i\n\r" % self.prior['dip']['range_max'])
        txt_file.write("dip_range_min:%i\n\r" % self.prior['dip']['range_min'])
        txt_file.write("angle_spread:%i\n\r" % self.prior['angle_spread'])
        txt_file.write("rake_range_max:%i\n\r" % self.prior['rake']['range_max'])
        txt_file.write("rake_range_min:%i\n\r" % self.prior['rake']['range_min'])
        txt_file.write("rake_spread:%i\n\r" % self.prior['rake']['spread'])
        txt_file.write("%i,%i,%i,%i,%i,%i\n\r" % (
            self.time_at_rec.year, self.time_at_rec.month, self.time_at_rec.day,
            self.time_at_rec.hour, self.time_at_rec.minute, self.time_at_rec.second))  #

    def write_sample(self, file_name, epi, depth, moment, Xi_old, s_z_old, s_r_old, s_t_old, p_z_old, p_r_old, bw_old,
                     R_dict, rw_old, L_dict, lw_old, accept=0):
        file_name.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f," % (
            epi, depth, moment[0], moment[1], moment[2], Xi_old, s_z_old, s_r_old, s_t_old, p_z_old, p_r_old, bw_old,
            rw_old, lw_old))
        for j, v in R_dict.iteritems():
            file_name.write("%.4f," % v)
        for j, v in L_dict.iteritems():
            file_name.write("%.4f," % v)
        file_name.write("%i,%i,%i\n\r" % (accept, len(R_dict), len(L_dict)))

    def run_parallel(self):
        if not self.MCMC == 'M' or self.MCMC == 'MH':
            raise ValueError('Choose MCMC algorithm, either: M or MH')

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        print("Rank", rank, "Size", size)
        dir_proc = self.directory + '/proc_sdr'

        if not os.path.exists(dir_proc):
            os.makedirs(dir_proc)
        filepath_proc = dir_proc + '/file_proc_%i.txt' % rank

        try:
            self.start_MCMC(filepath_proc)
        except TypeError:
            print ("TypeError in rank: %i" % rank)
            self.start_MCMC(filepath_proc)

        comm.bcast()  # All the processor wait until they are all at this point
        if rank == 0:
            print ("The .txt files are saved in: %s" % dir_proc)
