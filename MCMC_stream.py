import mpi4py as MPI
from mpi4py import MPI
import os
import numpy as np
import geographiclib.geodesic as geo
import seaborn as sns
import matplotlib.pylab as plt
import obspy

from Source_code import Source_code
from Seismogram import Seismogram
from Green_functions import Green_functions
from Inversion_problems import Inversion_problem
from Misfit import Misfit


class MCMC_stream:
    def __init__(self, total_traces_obs, P_traces_obs , S_traces_obs, PRIOR, db, specification_values, time_at_receiver):
        # specification values is a dict of the following, See Get_Parameters.py to see the form of values:
        self.sdr = specification_values['sdr']
        self.plot_modus = specification_values['plot_modus']
        self.xi = specification_values['misfit']
        self.MCMC = specification_values['MCMC']
        self.temperature = specification_values['temperature']
        self.directory = specification_values['directory']
        self.noise = specification_values['noise']
        self.npts = specification_values['npts']

        self.total_tr_obs = total_traces_obs
        self.s_tr_obs = S_traces_obs
        self.p_tr_obs = P_traces_obs
        self.prior = PRIOR
        self.db = db

        # V_stack your observed traces:
        self.d_obs = np.array([])
        for trace in self.total_tr_obs.traces:
            self.d_obs = np.append(self.d_obs, trace)

        # General parameters:
        self.dt = total_traces_obs[0].meta.delta
        self.temperature = specification_values['temperature']
        self.time_at_rec = time_at_receiver

        # Initiate all the python codes needed:
        self.window = Source_code(self.prior['VELOC_taup'])
        self.seis = Seismogram(self.prior, self.db)
        self.Green_function = Green_functions(self.prior, self.db)
        self.inv = Inversion_problem(self.prior)
        self.misfit = Misfit(specification_values['directory'])

    def model_samples(self, epi_old=None, depth_old=None):
        if epi_old == None or depth_old == None:
            epi_sample = np.random.uniform(self.prior['epi']['range_min'], self.prior['epi']['range_max'])
            depth_sample =np.around(np.random.uniform(self.prior['depth']['range_min'], self.prior['depth']['range_max']),decimals=1)
        else:
            epi_sample = np.random.normal(epi_old, self.prior['epi']['spread'])
            depth_sample = np.around(np.random.normal(depth_old, self.prior['depth']['spread']),decimals=1)
        return epi_sample, depth_sample

    def model_samples_sdr(self, strike_old=None, dip_old=None, rake_old=None):
        if strike_old == None or dip_old == None or rake_old == None:
            strike = np.random.uniform(0, 2 * np.pi, 1)  # Phi
            dip = np.arccos(1 - 2 * np.random.random_sample((1,))) # Theta
            rake = np.random.uniform(self.prior['rake']['range_min'], self.prior['rake']['range_max'])
        else:
            strike = np.random.normal(strike_old, self.prior['strike']['spread'])
            dip = np.random.normal(dip_old, self.prior['dip']['spread'])
            rake = np.random.normal(rake_old, self.prior['rake']['spread'])
        return 79,50,20

    def G_function(self, epi, depth, moment_old=None):
        if self.sdr == True:
            dict = geo.Geodesic(a=self.prior['radius'], f=0).ArcDirect(lat1=self.prior['la_r'], lon1=self.prior['lo_r'],
                                                         azi1=self.prior['baz'],
                                                         a12=epi, outmask=1929)
            if moment_old is None:
                strike, dip, rake = self.model_samples_sdr()
            else:
                strike, dip, rake = self.model_samples_sdr(moment_old[0], moment_old[1], moment_old[2])
                if strike < self.prior['strike']['range_min'] or strike > self.prior['strike']['range_max'] or dip < \
                        self.prior['dip']['range_min'] or dip > self.prior['dip']['range_max'] or rake < \
                        self.prior['rake']['range_min'] or rake > self.prior['rake']['range_max']:
                    total_syn = None
                    s_syn = None
                    p_syn = None
                    moment = None
                    return total_syn,p_syn, s_syn, moment
            d_syn, traces_syn, sources = self.seis.get_seis_manual(la_s=dict['lat2'], lo_s=dict['lon2'], depth=depth,
                                                                   strike=strike, dip=dip, rake=rake,
                                                                   time=self.time_at_rec, sdr=self.sdr)

            total_syn, p_syn, s_syn = self.window.get_window_obspy(traces_syn, epi, depth, self.time_at_rec,self.npts)

            return total_syn,p_syn, s_syn, np.array([strike, dip, rake])

        else:
            G_tot, G_z, G_r, G_t = self.Green_function.get(self.time_at_rec,epi, depth, self.npts)
            moment = self.inv.Solve_damping_smoothing(self.d_obs, G_tot)
            d_syn = np.matmul(G_tot,moment)
            return d_syn, moment

    def start_MCMC(self,savepath):
        with open(savepath, 'w') as save_file:
            self.write_par(save_file)
            accepted = 0
            rejected = 0
            for i in range(self.prior['sample_number']):
                if i % 10 == 0:
                    print("proposal: %i, accepted: %i" % (i, accepted))
                    print("proposal: %i, rejected: %i" % (i, rejected))

                if i ==0 or self.MCMC == 'MH':
                    epi,depth = self.model_samples()
                    total_syn, p_syn, s_syn, moment = self.G_function(epi, depth)
                else:
                    epi, depth = self.model_samples(epi_old, depth_old)

                    if epi < self.prior['epi']['range_min'] or epi > self.prior['epi']['range_max'] or depth < \
                            self.prior['depth']['range_min'] or depth > self.prior['depth']['range_max']:
                        continue
                    total_syn, p_syn, s_syn, moment = self.G_function(epi, depth, moment_old)
                    if p_syn is None or moment is None:
                        continue

                if self.xi == 'L2':
                    Xi_new, time_shift_new = self.misfit.L2_stream(self.p_tr_obs,p_syn,self.s_tr_obs,s_syn,self.time_at_rec,self.prior['var_est'])
                elif self.xi == 'CC':
                    Xi_new, time_shift_new = self.misfit.CC_stream(self.p_tr_obs,p_syn,self.s_tr_obs,s_syn,self.time_at_rec)
                else:
                  raise ValueError('misfit is not specified correctly, choose either: L2 or CC in string format')

                if i == 0:
                    self.write_sample(save_file, epi, depth, Xi_new, moment)
                    Xi_old = Xi_new
                    epi_old = epi
                    depth_old = depth
                    moment_old = moment
                    if self.plot_modus == True:
                        sns.set()
                        f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
                        ax1.plot(total_syn[0], label="old_Z_syn")
                        ax2.plot(total_syn[1], label="old_R_syn")
                        ax3.plot(total_syn[2], label="old_T_syn")
                    continue

                random = np.random.random_sample((1,))
                if Xi_new < Xi_old or np.exp((Xi_old - Xi_new) / self.temperature) > random:

                    if self.plot_modus == True:
                        ax1.plot(total_syn[0])
                        ax2.plot(total_syn[1])
                        ax3.plot(total_syn[2])
                    self.write_sample(save_file, epi, depth, Xi_new, moment)
                    epi_old = epi
                    depth_old = depth
                    Xi_old = Xi_new
                    moment_old = moment
                    accepted = accepted + 1
                else:
                    rejected += 1

                    # self.write_sample(save_file, epi_old, depth_old, Xi_old, moment_old, sdr)
            if self.plot_modus == True:
                ax1.plot(self.total_tr_obs[0], linestyle=':', label="Observed data")
                ax2.plot(self.total_tr_obs[1], linestyle=':', label="Observed data")
                ax3.plot(self.total_tr_obs[2], linestyle=':', label="Observed data")
                ax1.legend()
                ax2.legend()
                ax3.legend()
                plt.tight_layout()
                plt.savefig(savepath.replace('.txt', '.pdf'))
                plt.close()
        save_file.close()

    def write_par(self, txt_file):
        txt_file.write("%s\n\r" % self.prior['VELOC'])  # Velocity model used
        txt_file.write("%s\n\r" % self.MCMC)  # Velocity model used
        txt_file.write("%s\n\r" % self.xi)  # Velocity model used
        txt_file.write("%r\n\r" % self.noise)  # Velocity model used
        txt_file.write("%r\n\r" % self.sdr)  # Velocity model used
        txt_file.write("%r\n\r" % self.plot_modus)  # Velocity model used
        # txt_file.write("%.4f\n\r" % self.par['MO'])  #
        txt_file.write("%.4f\n\r" % self.prior['alpha'])  #
        txt_file.write("%.4f\n\r" % self.prior['beta'])  #
        txt_file.write("%.4f\n\r" % self.prior['az'])  #
        txt_file.write(
            "%s,%s,%s\n\r" % (self.prior['components'][0], self.prior['components'][1], self.prior['components'][2]))  #
        txt_file.write("%.4f\n\r" % self.prior['la_r'])  #
        txt_file.write("%.4f\n\r" % self.prior['lo_r'])  #
        txt_file.write("%s\n\r" % self.prior['filter'])  #
        txt_file.write("%s\n\r" % self.prior['definition'])  #
        txt_file.write("%s\n\r" % self.prior['kind'])  #
        txt_file.write("%s\n\r" % self.prior['network'])  #
        txt_file.write("%i\n\r" % self.prior['sample_number'])  #
        txt_file.write("%.4f\n\r" % self.prior['var_est'])  #
        txt_file.write("%i\n\r" % self.prior['epi']['range_max'])  #
        txt_file.write("%i\n\r" % self.prior['epi']['range_min'])  #
        txt_file.write("%i\n\r" % self.prior['epi']['spread'])  #
        txt_file.write("%i\n\r" % self.prior['depth']['range_max'])  #
        txt_file.write("%i\n\r" % self.prior['depth']['range_min'])  #
        txt_file.write("%i\n\r" % self.prior['depth']['spread'])
        txt_file.write("%i\n\r" % self.prior['strike']['range_max'])
        txt_file.write("%i\n\r" % self.prior['strike']['range_min'])
        txt_file.write("%i\n\r" % self.prior['strike']['spread'])
        txt_file.write("%i\n\r" % self.prior['dip']['range_max'])
        txt_file.write("%i\n\r" % self.prior['dip']['range_min'])
        txt_file.write("%i\n\r" % self.prior['dip']['spread'])
        txt_file.write("%i\n\r" % self.prior['rake']['range_max'])
        txt_file.write("%i\n\r" % self.prior['rake']['range_min'])
        txt_file.write("%i\n\r" % self.prior['rake']['spread'])
        txt_file.write("%i,%i,%i,%i,%i,%i\n\r" % (
            self.time_at_rec.year, self.time_at_rec.month, self.time_at_rec.day,
            self.time_at_rec.hour, self.time_at_rec.minute, self.time_at_rec.second))  #

    def write_sample(self, file_name, epi, depth, Xi, moment):
        if self.sdr == True:
            file_name.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n\r" % (epi, depth, moment[0], moment[1],
                                                                   moment[2], Xi))
        else:
            file_name.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n\r" % (
                epi, depth, moment[0], moment[1], moment[2], moment[3], moment[4], Xi))

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
        self.start_MCMC(filepath_proc)

        comm.bcast()  # All the processor wait until they are all at this point
        if rank == 0:
            print ("The .txt files are saved in: %s" % dir_proc)
