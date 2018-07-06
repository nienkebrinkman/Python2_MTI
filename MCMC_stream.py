import mpi4py as MPI
from mpi4py import MPI
import os
import numpy as np
import geographiclib.geodesic as geo
import seaborn as sns
import matplotlib.pylab as plt
import instaseis

from Create_observed import Create_observed
from Get_Parameters import Get_Paramters
from Source_code import Source_code
from Seismogram import Seismogram
from Green_functions import Green_functions
from Inversion_problems import Inversion_problem
from Misfit import Misfit
from Surface_waves import Surface_waves


class MCMC_stream:
    def __init__(self, R_env_obs, L_env_obs, total_traces_obs, P_traces_obs, S_traces_obs, PRIOR, db,
                 specification_values,
                 time_at_receiver,start_sample_path = None):
        # specification values is a dict of the following, See Get_Parameters.py to see the form of values:
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

        # General parameters:
        self.dt = total_traces_obs[0].meta.delta
        self.temperature = specification_values['temperature']
        self.time_at_rec = time_at_receiver
        self.start_sample = start_sample_path
        self.rnd_par = specification_values['rnd_par']

        # Initiate all the python codes needed:
        self.window = Source_code(self.prior['VELOC_taup'])
        self.seis = Seismogram(self.prior, self.db)
        self.Green_function = Green_functions(self.prior, self.db)
        self.inv = Inversion_problem(self.prior)
        self.misfit = Misfit(specification_values['directory'])
        self.SW = Surface_waves(self.prior)

        # V_stack your observed traces:

        # If you uncomment this also change the trace array you enter in the code!!!

        # self.d_obs_stream= self.window.stack_BW_SW_Streams(self.total_tr_obs,self.R_env_obs,self.L_env_obs)
        # self.d_obs_array=np.array([])
        # for i in self.d_obs_stream.traces:
        #     self.d_obs_array = np.append(self.d_obs_array,i.data)

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
                epi_sample = np.random.normal(epi_old, self.prior['epi']['spread'])
                depth_sample = np.around(np.random.normal(depth_old, self.prior['depth']['spread']), decimals=1)
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
        if self.sdr == True:
            if moment_old is None:
                strike, dip, rake = self.model_samples_sdr()
            else:
                strike, dip, rake = self.model_samples_sdr(moment_old[0], moment_old[1], moment_old[2])
            d_syn, traces_syn, sources = self.seis.get_seis_manual(la_s=dict['lat2'], lo_s=dict['lon2'], depth=depth,
                                                                   strike=strike, dip=dip, rake=rake,
                                                                   time=self.time_at_rec, sdr=self.sdr)
            R_env_syn = self.SW.rayleigh_pick(Z_trace=traces_syn.traces[0], la_s=dict['lat2'], lo_s=dict['lon2'],
                                              depth=depth, save_directory=self.directory, time_at_rec=self.time_at_rec,
                                              npts=self.npts, plot_modus=False)
            L_env_syn = self.SW.love_pick(T_trace=traces_syn.traces[2], la_s=dict['lat2'], lo_s=dict['lon2'],
                                          depth=depth, save_directory=self.directory, time_at_rec=self.time_at_rec,
                                          npts=self.npts, plot_modus=False)

            total_syn, p_syn, s_syn = self.window.get_window_obspy(traces_syn, epi, depth, self.time_at_rec, self.npts)

            return R_env_syn, L_env_syn, total_syn, p_syn, s_syn, np.array([strike, dip, rake])

        else:
            G_z_r_t, G_z, G_r, G_t = self.Green_function.get_bw(self.time_at_rec, epi, depth, self.npts)
            G_R_L, G_R, G_L = self.Green.get_sw(self.time_at_rec, epi, depth, dict['lat2'], dict['lon2'], self.npts)
            G_total = np.vstack((G_z_r_t, G_R_L))

            moment = self.inv.Solve_LS(self.d_obs_array, G_total)
            d_syn = np.matmul(G_total, moment)
            d_syn_split = self.window.split_traces(d_syn, self.d_obs_stream, self.time_at_rec)
            total_syn, p_syn, s_syn, R_syn, L_syn = self.window.split_BW_SW(d_syn_split, epi, depth, self.time_at_rec,
                                                                            self.npts)

            return total_syn, p_syn, s_syn, moment

    def start_MCMC(self, savepath):
        with open(savepath, 'w') as save_file:
            self.write_par(save_file)
            accepted = 0
            rejected = 0
            for i in range(self.prior['sample_number']):
                if i % 10 == 0:
                    print("proposal: %i, accepted: %i" % (i, accepted))
                    print("proposal: %i, rejected: %i" % (i, rejected))

                if i == 0 or self.MCMC == 'MH':
                    if self.start_sample == None:
                        epi, depth = self.model_samples()
                        R_env_syn, L_env_syn, total_syn, p_syn, s_syn, moment = self.G_function(epi, depth)
                    else:
                        if self.rnd_par == True:
                            self.update = np.random.choice(['epi', 'depth', 'moment'], 1)[0]
                        else:
                            self.update = None
                        data = np.loadtxt(self.start_sample, delimiter=',')
                        epi = data[0]
                        depth = data[1]
                        moment_old = np.array([data[2], data[3], data[4]])
                        R_env_syn, L_env_syn, total_syn, p_syn, s_syn, moment = self.G_function(epi, depth, moment_old)

                else:
                    if self.rnd_par == True:
                        self.update = np.random.choice(['epi','depth','moment'],1)[0]
                    else:
                        self.update = None
                    epi, depth = self.model_samples(epi_old, depth_old)
                    if epi < self.prior['epi']['range_min'] or epi > self.prior['epi']['range_max'] or depth < \
                            self.prior['depth']['range_min'] or depth > self.prior['depth']['range_max']:
                        continue
                    R_env_syn, L_env_syn, total_syn, p_syn, s_syn, moment = self.G_function(epi, depth, moment_old)


                if self.xi == 'L2':
                    Xi_bw, time_shift_new = self.misfit.L2_stream(self.p_tr_obs, p_syn, self.s_tr_obs, s_syn,
                                                                  self.time_at_rec, self.prior['var_est'])
                    Xi_R = self.misfit.SW_L2(self.R_env_obs, R_env_syn, self.prior['var_est'])
                    Xi_L = self.misfit.SW_L2(self.L_env_obs, L_env_syn, self.prior['var_est'])
                    Xi_new = Xi_bw + Xi_R + Xi_L
                elif self.xi == 'CC':
                    Xi_bw, time_shift_new = self.misfit.CC_stream(self.p_tr_obs, p_syn, self.s_tr_obs, s_syn,
                                                                  self.time_at_rec)
                    Xi_R = self.misfit.SW_L2(self.R_env_obs, R_env_syn, self.prior['var_est'])
                    Xi_L = self.misfit.SW_L2(self.L_env_obs, L_env_syn, self.prior['var_est'])
                    Xi_new = Xi_bw + Xi_R + Xi_L
                else:
                    raise ValueError('misfit is not specified correctly, choose either: L2 or CC in string format')

                if i == 0:
                    Xi_old = Xi_new
                    epi_old = epi
                    depth_old = depth
                    moment_old = moment
                    self.write_sample(save_file, epi, depth, Xi_new, Xi_new, moment, epi,depth,moment, accept=1)
                    if self.plot_modus == True:
                        sns.set()
                        plt.figure(9)
                        plt.subplot(311)
                        plt.plot(total_syn[0], label="old_Z_syn")
                        plt.subplot(312)
                        plt.plot(total_syn[1], label="old_R_syn")
                        plt.subplot(313)
                        plt.plot(total_syn[2], label="old_T_syn")
                        # f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
                        # ax1.plot(total_syn[0], label="old_Z_syn")
                        # ax2.plot(total_syn[1], label="old_R_syn")
                        # ax3.plot(total_syn[2], label="old_T_syn")
                    continue

                random = np.random.random_sample((1,))
                if Xi_new < Xi_old or np.exp((Xi_old - Xi_new) / self.temperature) > random:

                    if self.plot_modus == True:
                        plt.figure(9)
                        plt.subplot(311)
                        plt.plot(total_syn[0])
                        plt.subplot(312)
                        plt.plot(total_syn[1])
                        plt.subplot(313)
                        plt.plot(total_syn[2])
                        # ax1.plot(total_syn[0])
                        # ax2.plot(total_syn[1])
                        # ax3.plot(total_syn[2])
                    self.write_sample(save_file, epi, depth, Xi_new, Xi_new, moment, epi, depth,moment ,accept=1)
                    epi_old = epi
                    depth_old = depth
                    Xi_old = Xi_new
                    moment_old = moment
                    accepted = accepted + 1
                else:
                    rejected += 1
                    self.write_sample(save_file, epi_old, depth_old, Xi_old, Xi_new, moment_old, epi,depth,moment,accept=0)
                    if self.plot_modus == True:
                        plt.figure(10)
                        plt.subplot(311)
                        plt.plot(total_syn[0])
                        plt.subplot(312)
                        plt.plot(total_syn[1])
                        plt.subplot(313)
                        plt.plot(total_syn[2])

            if self.plot_modus == True:
                plt.figure(9)
                plt.subplot(311)
                plt.plot(self.total_tr_obs[0], linestyle=':', label="Observed data")
                plt.legend(loc='upper right')
                plt.tight_layout()
                plt.subplot(312)
                plt.plot(self.total_tr_obs[1], linestyle=':', label="Observed data")
                plt.legend(loc='upper right')
                plt.tight_layout()
                plt.subplot(313)
                plt.plot(self.total_tr_obs[2], linestyle=':', label="Observed data")
                plt.legend(loc='upper right')
                # ax1.plot(self.total_tr_obs[0].data[:2000], linestyle=':', label="Observed data")
                # ax2.plot(self.total_tr_obs[1].data[:2000], linestyle=':', label="Observed data")
                # ax3.plot(self.total_tr_obs[2].data[:2000], linestyle=':', label="Observed data")
                # ax1.legend()
                # ax2.legend()
                # ax3.legend()
                plt.tight_layout()
                plt.savefig(self.directory + '/accepted.pdf')
                plt.close()

                plt.figure(10)
                plt.subplot(311)
                plt.plot(self.total_tr_obs[0], linestyle=':', label="Observed data")
                plt.legend(loc='upper right')
                plt.tight_layout()
                plt.subplot(312)
                plt.plot(self.total_tr_obs[1], linestyle=':', label="Observed data")
                plt.legend(loc='upper right')
                plt.tight_layout()
                plt.subplot(313)
                plt.plot(self.total_tr_obs[2], linestyle=':', label="Observed data")
                plt.legend(loc='upper right')
                # ax1.plot(self.total_tr_obs[0].data[:2000], linestyle=':', label="Observed data")
                # ax2.plot(self.total_tr_obs[1].data[:2000], linestyle=':', label="Observed data")
                # ax3.plot(self.total_tr_obs[2].data[:2000], linestyle=':', label="Observed data")
                # ax1.legend()
                # ax2.legend()
                # ax3.legend()
                plt.tight_layout()
                plt.savefig(self.directory + '/rejected.pdf')
                plt.close()
        save_file.close()

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

    def write_sample(self, file_name, epi, depth, Xi_new, Xi_old, moment, epi_old, depth_old, moment_old, accept=0):
        if self.sdr == True:
            file_name.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%i,%.4f,%.4f,%.4f,%.4f,%.4f\n\r" % (
                epi, depth, moment[0], moment[1],
                moment[2], Xi_new, Xi_old, accept, epi_old, depth_old, moment_old[0], moment_old[1],
                moment_old[2]))
        else:
            file_name.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f%.4f,%.4f,%i,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n\r" % (
                epi, depth, moment[0], moment[1], moment[2], moment[3], moment[4], Xi_new, Xi_old, accept, epi_old,
                depth_old, moment_old[0], moment_old[1], moment_old[2], moment_old[3], moment_old[4]))

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
