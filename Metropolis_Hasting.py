import numpy as np
import obspy
import os
import yaml
import mpi4py as MPI
import multiprocessing as mp

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
        year = self.par['origin_time'].year # Constant
        month = self.par['origin_time'].month # Constant
        day = self.par['origin_time'].day # Constant
        hour = int(np.random.uniform(self.sampler['time_range'], self.par['origin_time'].hour+1))
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
        ## -- choose a range for moment with the help of the Resolution Matrix --
        ## NOT FINISHED YET!!!
        d_syn = np.matmul(G, moment)
        return d_syn

    def do(self):
        accept = {'epi': np.array([]), 'azimuth': np.array([]), 'depth': np.array([]),
                  'time': np.array([]), 'misfit': np.array([])}
        ## Starting parameters and create A START MODEL (MODEL_OLD):
        epi, azimuth, depth, time = self.model_samples()
        d_syn_old = self.G_function(epi, depth, azimuth, time)
        misfit = Misfit()
        Xi_old = misfit.get_xi(self.d_obs, d_syn_old, self.sampler['var_est'])
        accept['epi'] = np.append(accept['epi'], epi)
        accept['azimuth'] = np.append(accept['azimuth'], azimuth)
        accept['depth'] = np.append(accept['depth'], depth)
        accept['time']=np.append(accept['time'],time.timestamp)
        accept['misfit'] = np.append(accept['misfit'], Xi_old)

        for i in range(self.sampler['sample_number']):
            epi, azimuth, depth, time = self.model_samples()
            d_syn = self.G_function(epi, depth, azimuth, time)
            misfit = Misfit()
            Xi_new = misfit.get_xi(self.d_obs, d_syn, self.sampler['var_est'])
            random = np.random.random_sample((1,))
            if Xi_new < Xi_old or (Xi_old / Xi_new) > random:
                accept['epi'] = np.append(accept['epi'], epi)
                accept['azimuth'] = np.append(accept['azimuth'], azimuth)
                accept['depth'] = np.append(accept['depth'], depth)
                accept['time']=np.append(accept['time'],time.timestamp)
                accept['misfit'] = np.append(accept['misfit'], Xi_new)
                Xi_old = Xi_new
            else:
                accept['epi'] = np.append(accept['epi'], accept['epi'][i])
                accept['azimuth'] = np.append(accept['azimuth'], accept['azimuth'][i])
                accept['depth'] = np.append(accept['depth'], accept['depth'][i])
                accept['time'] = np.append(accept['time'],accept['time'][i])
                accept['misfit'] = np.append(accept['misfit'], Xi_old)
                continue
        filepath = self.sampler['filepath']
        if os.path.isfile(filepath) == True:
            filename_new = 'NEW'
            print('The filename already exists: it is now saved as: %s ' %filename_new)
            path = os.path.join(self.par['directory'], filename_new)
            with open(path, 'w') as yaml_file:
                yaml.dump(accept, yaml_file, default_flow_style=False)
            yaml_file.close()
            return accept
        else:
            with open(filepath, 'w') as yaml_file:
                yaml.dump(accept, yaml_file, default_flow_style=False)
            yaml_file.close()
            return accept

#-----------------------------------------------------------------------------------------------------------------------
# Run parallel [NOT FINISHED YET]

    def do_sampler(self):
        accept = {'epi': np.array([]), 'azimuth': np.array([]), 'depth': np.array([]),
                  'time': np.array([]), 'misfit': np.array([])}
        ## Starting parameters and create A START MODEL (MODEL_OLD):
        epi, azimuth, depth, time = self.model_samples()
        d_syn_old = self.G_function(epi, depth, azimuth, time)
        misfit = Misfit()
        Xi_old = misfit.get_xi(self.d_obs, d_syn_old, self.sampler['var_est'])
        accept['epi'] = np.append(accept['epi'], epi)
        accept['azimuth'] = np.append(accept['azimuth'], azimuth)
        accept['depth'] = np.append(accept['depth'], depth)
        accept['time'] = np.append(accept['time'], time.timestamp)
        accept['misfit'] = np.append(accept['misfit'], Xi_old)

        for i in range(self.sampler['sample_number']):
            epi, azimuth, depth, time = self.model_samples()
            d_syn = self.G_function(epi, depth, azimuth, time)
            misfit = Misfit()
            Xi_new = misfit.get_xi(self.d_obs, d_syn, self.sampler['var_est'])
            random = np.random.random_sample((1,))
            if Xi_new < Xi_old or (Xi_old / Xi_new) > random:
                accept['epi'] = np.append(accept['epi'], epi)
                accept['azimuth'] = np.append(accept['azimuth'], azimuth)
                accept['depth'] = np.append(accept['depth'], depth)
                accept['time'] = np.append(accept['time'], time.timestamp)
                accept['misfit'] = np.append(accept['misfit'], Xi_new)
                Xi_old = Xi_new
            else:
                accept['epi'] = np.append(accept['epi'], accept['epi'][i])
                accept['azimuth'] = np.append(accept['azimuth'], accept['azimuth'][i])
                accept['depth'] = np.append(accept['depth'], accept['depth'][i])
                accept['time'] = np.append(accept['time'], accept['time'][i])
                accept['misfit'] = np.append(accept['misfit'], Xi_old)
                continue
        self.output.put((accept))

    def do_parallel(self):
        self.output = mp.Queue()

        # Setup a list of processes that we want to run
        processes = [mp.Process(target=self.do_sampler) for x in range(4)]

        # Run processes
        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()

        # Get process results from the output queue
        results = [self.output.get() for p in processes]

        print(results)




