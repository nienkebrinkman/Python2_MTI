import numpy as np
import obspy
import os
import yaml
import mpi4py as MPI
# import multiprocessing
from mpi4py import MPI

from Inversion_problems import Inversion_problem
from Misfit import Misfit


class MH_algorithm:
    def __init__(self, PARAMETERS, sampler, db, data):
        self.db = db
        self.par = PARAMETERS
        self.sampler = sampler
        self.d_obs = data

    def model_samples(self):
        epi_sample = np.random.uniform(self.sampler['epi']['range_min'], self.sampler['epi']['range_max'])
        azimuth_sample = np.random.uniform(self.sampler['azimuth']['range_min'], self.sampler['azimuth']['range_max'])
        depth_sample = np.random.uniform(self.sampler['depth']['range_min'], self.sampler['depth']['range_max'])

        # Time sampler:
        year = self.par['origin_time'].year  # Constant
        month = self.par['origin_time'].month  # Constant
        day = self.par['origin_time'].day  # Constant
        hour = int(np.random.uniform(self.sampler['time_range'], self.par['origin_time'].hour + 1))
        min = int(np.random.uniform(0, 60))
        sec = int(np.random.uniform(1, 60))
        time_sample = obspy.UTCDateTime(year, month, day, hour, min, sec)

        return epi_sample, azimuth_sample, depth_sample, time_sample

    def generate_G(self, epi, depth, azimuth, t):
        gf = self.db.get_greens_function(epicentral_distance_in_degree=epi, source_depth_in_m=depth, origin_time=t,
                                         kind=self.par['kind'], kernelwidth=self.par['kernelwidth'],
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

    def G_function(self, epi, depth, azimuth, t):
        G = self.generate_G(epi, depth, azimuth, t)
        inv = Inversion_problem(self.d_obs, G, self.par)
        moment = inv.Solve_damping_smoothing()
        # TODO - choose a range for moment with the help of the Resolution Matrix
        d_syn = np.matmul(G, moment)
        return d_syn, moment

# ---------------------------------------------------------------------------------------------------------------------#
#                                                 NOT parallel                                                         #
# ---------------------------------------------------------------------------------------------------------------------#


    def do(self):
        with open(self.sampler['filepath'], 'w') as yaml_file:
            ## Starting parameters and create A START MODEL (MODEL_OLD):
            epi_old, azimuth_old, depth_old, time_old = self.model_samples()
            d_syn_old, moment_old = self.G_function(epi_old, depth_old, azimuth_old, time_old)
            misfit = Misfit()
            Xi_old = misfit.get_xi(self.d_obs, d_syn_old, self.sampler['var_est'])
            yaml_file.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4r,%.4f,%.4f\n\r" % (
                epi_old, azimuth_old, depth_old, time_old.timestamp, Xi_old, moment_old[0], moment_old[1],
                moment_old[2], moment_old[3], moment_old[4]))

            for i in range(self.sampler['sample_number']):
                epi, azimuth, depth, time = self.model_samples()
                d_syn, moment = self.G_function(epi, depth, azimuth, time)
                misfit = Misfit()
                Xi_new = misfit.get_xi(self.d_obs, d_syn, self.sampler['var_est'])
                random = np.random.random_sample((1,))
                if Xi_new < Xi_old or (Xi_old / Xi_new) > random:
                    yaml_file.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4r,%.4f,%.4f\n\r" % (
                        epi, azimuth, depth, time.timestamp, Xi_new, moment[0], moment[1], moment[2], moment[3],
                        moment[4]))
                    Xi_old = Xi_new
                    epi_old = epi
                    azimuth_old = azimuth
                    depth_old = depth
                    time_old = time
                    moment_old = moment
                else:
                    yaml_file.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4r,%.4f,%.4f\n\r" % (
                        epi_old, azimuth_old, depth_old, time_old.timestamp, Xi_old, moment_old[0], moment_old[1],
                        moment_old[2], moment_old[3], moment_old[4]))
                    continue
        yaml_file.close()

# ---------------------------------------------------------------------------------------------------------------------#
#                                                 RUN parallel                                                         #
# ---------------------------------------------------------------------------------------------------------------------#
    def do_parallel(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        print("Rank", rank, "Size", size)
        self.processing(rank, size)

    def processing(self, rank, size):
        dir_proc = self.sampler['directory'] + '/proc'
        if not os.path.exists(dir_proc):
            os.makedirs(dir_proc)
        filepath_proc = dir_proc + '/file_proc_%i.txt' % rank

        with open(filepath_proc, 'w') as yaml_file:
            ## Starting parameters and create A START MODEL (MODEL_OLD):
            epi_old, azimuth_old, depth_old, time_old = self.model_samples()
            d_syn_old, moment_old = self.G_function(epi_old, depth_old, azimuth_old, time_old)
            misfit = Misfit()
            Xi_old = misfit.get_xi(self.d_obs, d_syn_old, self.sampler['var_est'])
            yaml_file.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4r,%.4f,%.4f\n\r" % (
                epi_old, azimuth_old, depth_old, time_old.timestamp, Xi_old, moment_old[0], moment_old[1],
                moment_old[2], moment_old[3], moment_old[4]))

            for i in range(self.sampler['sample_number']):
                epi, azimuth, depth, time = self.model_samples()
                d_syn, moment = self.G_function(epi, depth, azimuth, time)
                misfit = Misfit()
                Xi_new = misfit.get_xi(self.d_obs, d_syn, self.sampler['var_est'])
                random = np.random.random_sample((1,))
                if Xi_new < Xi_old or (Xi_old / Xi_new) > random:
                    yaml_file.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4r,%.4f,%.4f\n\r" % (
                        epi, azimuth, depth, time.timestamp, Xi_new, moment[0], moment[1], moment[2], moment[3],
                        moment[4]))
                    Xi_old = Xi_new
                    epi_old = epi
                    azimuth_old = azimuth
                    depth_old = depth
                    time_old = time
                    moment_old = moment
                else:
                    yaml_file.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4r,%.4f,%.4f\n\r" % (
                        epi_old, azimuth_old, depth_old, time_old.timestamp, Xi_old, moment_old[0], moment_old[1],
                        moment_old[2], moment_old[3], moment_old[4]))
                    continue
        yaml_file.close()
