import instaseis
import obspy
import numpy as np
import matplotlib.pyplot as plt
from obspy.imaging.beachball import beachball
import pylab
from obspy.geodetics import kilometer2degrees
from obspy.geodetics.base import gps2dist_azimuth
from obspy.taup import TauPyModel
import os
import os.path
import yaml



## Velocity model:
VELOC = 'http://instaseis.ethz.ch/marssynthetics/C80VL-BFT13-1s'
VELOC_taup = 'iasp91'

# Safe parameters:
directory = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/testdata'
filename = 'test_2.yaml'
filepath = os.path.join(directory, filename)



## Parameters:
PARAMETERS = {
    'la_s': 'la_s',
    'la_r': 'la_r',
    'lo_s': 'lo_s',
    'network': 'network',
    'strike': 'strike',
    'dip': 'dip',
    'rake': 'rake',
    'm_rr': 'm_rr',
    'm_tt': 'm_tt',
    'm_pp': 'm_pp',
    'm_rt': 'm_rt',
    'm_rp': 'm_rp',
    'm_tp': 'm_tp',
    'origin_time': 'origin_time',
    'filter': 'filter',
    'freq_filter': 'freq_filter',
    'az ': 'az',
    'epi': 'epi',
    'kind': 'displacement',
    'kernelwidth': 'kernelwidth',
    'definition': 'definition',
    'components': 'components',
    'alpha': 'alpha',
    'beta': 'beta',
    'm_ref': 'm_ref',
    'directory': directory,
    'filename': filename,
    'filepath': filepath}

# -Receiver
PARAMETERS['la_r'] = 40.0  # Latitude
PARAMETERS['lo_r'] = 20.0  # Longitude
PARAMETERS['network'] = "7J"  # Network
PARAMETERS['station'] = "SYNT1"  # Station

# -Source
PARAMETERS['la_s'] = 10.0
PARAMETERS['lo_s'] = 12.0
PARAMETERS['depth_s'] = 1000
PARAMETERS['strike'] = 79
PARAMETERS['dip'] = 10
PARAMETERS['rake'] = 20
PARAMETERS['M0'] = 1E17
# PARAMETERS['m_tt']=1.810000e+22
# PARAMETERS['m_pp']=-1.740000e+24
# PARAMETERS['m_rr']=1.710000e+24
# PARAMETERS['m_tp']=-1.230000e+24
# PARAMETERS['m_rt']=1.990000e+23
# PARAMETERS['m_rp']=-1.050000e+23
PARAMETERS['origin_time'] = obspy.UTCDateTime(2020, 1, 2, 3, 4, 5)
PARAMETERS['components'] = ["Z", "R", "T"]

# -filter
PARAMETERS['filter'] = 'highpass'
PARAMETERS['freq_filter'] = 1.0

# -Greens function
dist, az, baz = gps2dist_azimuth(lat1=PARAMETERS['la_s'],
                                 lon1=PARAMETERS['lo_s'],
                                 lat2=PARAMETERS['la_r'],
                                 lon2=PARAMETERS['lo_r'], a=3389.5, f=0)
PARAMETERS['az'] = az
PARAMETERS['epi'] = kilometer2degrees(dist, radius=3389.5)
PARAMETERS['kind'] = 'displacement'
PARAMETERS['kernelwidth'] = 12
PARAMETERS['definition'] = 'seiscomp'

# -Inversion parameters
PARAMETERS['alpha'] = 10 ** (-23)
PARAMETERS['beta'] = 10 ** (-23)
PARAMETERS['m_ref'] = np.array([1.0000e+16, 1.0000e+16, 1.0000e+16, 1.0000e+16, 1.0000e+16])

## - Sampler:

sampler = {
    'depth': {'range_min': 'min', 'range_max': 'max', 'step': 'step'},
    'epi': {'range_min': 'min', 'range_max': 'max', 'step': 'step'},
    'azimuth': {'range_min': 'min', 'range_max': 'max', 'step': 'step'},
    'time_range': 'time_range',
    'moment_range': 'moment_range',
    'sample_number': 'sample_number',
    'var_est': 'var_est'}
sampler['depth']['range_min'] = 800
sampler['depth']['range_max'] = 1200
sampler['depth']['step'] = 20
sampler['epi']['range_min'] = 20
sampler['epi']['range_max'] = 40
sampler['epi']['step'] = 40
sampler['azimuth']['range_min'] = 2
sampler['azimuth']['range_max'] = 22
sampler['azimuth']['step'] = 22
sampler['time_range'] = PARAMETERS['origin_time'].hour - 2  # The time range can only vary may be two hours!!
sampler['sample_number'] = 100000
sampler['var_est'] = 0.05


def main():
    ## Obtain database to create both Seismograms and Greens_functions:
    db = instaseis.open_db(VELOC)

    ## Make seismogram:
    seis = Seismogram(PARAMETERS, db)
    u, traces, source = seis.get()  # u = stacked seismograms , traces = 3 component seismogram separated

    ## Metropolis Hasting Algorithm
    # MH = MH_algorithm(PARAMETERS, sampler, db, u)
    # accept_model = MH.do(sampler['sample_number'])
    # MH.make_PDF()

    ## Get Green functions:
    Green = Green_functions(PARAMETERS, db)
    G = Green.get()

    ## Obtain Seismogram and Green function with certain window
    source_inv = Source_code(PARAMETERS, db)
    G_window, u_window = source_inv.get_windows(traces, G)

    ## Solve forward model:
    # moment_init = np.array([PARAMETERS['m_tt']+PARAMETERS['m_pp'], PARAMETERS['m_rr'], PARAMETERS['m_tp'], PARAMETERS['m_rt'],
    #           PARAMETERS['m_rp']])
    moment_init = np.array([source.m_tt, source.m_pp, -source.m_tp, source.m_rt,
                            -source.m_rp])
    print('Initial moment: \n%s' % moment_init)
    forward = Forward_problem(PARAMETERS, G, moment_init)
    data = forward.Solve_forward()

    Resolution_matrix = np.matmul(np.linalg.pinv(G), G)
    sampler['moment_range'] = Resolution_matrix


    ## Solve inversion method:
    inverse = Inversion_problem(u, G, PARAMETERS)
    moment_d = inverse.Solve_damping()
    moment_ds = inverse.Solve_damping_smoothing()
    moment_svd = inverse.Solve_SVD()

class Source_code:
    def __init__(self, PARAMETERS, db):
        self.par = PARAMETERS
        self.db = db

    def get_P(self):
        model = TauPyModel(model=VELOC_taup)
        tt = model.get_travel_times(source_depth_in_km=self.par['depth_s'] / 1000, distance_in_degree=self.par['epi'],
                                    phase_list=['P'], receiver_depth_in_km=0.0)
        return tt[0].time

    def get_S(self):
        model = TauPyModel(model=VELOC_taup)
        tt = model.get_travel_times(source_depth_in_km=self.par['depth_s'] / 1000, distance_in_degree=self.par['epi'],
                                    phase_list=['S'], receiver_depth_in_km=0.0)
        return tt[0].time

    def get_windows(self, traces, G):
        tt_P = self.get_P()  # Estimated P-wave arrival, based on the known velocity model
        tt_S = self.get_S()  # Estimated S-wave arrival, based on the known velocity model
        sec_per_sample = 1 / (traces[0].meta.sampling_rate)

        new_trace = np.array([])
        for i, trace in enumerate(traces.traces):
            sample_P = int(tt_P / sec_per_sample)
            sample_S = int(tt_S / sec_per_sample)
            ## Define window for P-arrival:
            tt_diff = np.abs(sample_S - sample_P)
            sample_P_min = int(sample_P - tt_diff * 0.5)
            sample_P_max = int(sample_P + tt_diff * 0.5 + 1)
            p_array = trace[sample_P_min:sample_P_max]
            new_trace = np.append(new_trace, p_array)

            ## Define window for S-arrival:
            sample_S_min = int(sample_S - tt_diff * 0.5)
            sample_S_max = int(sample_S + tt_diff * 0.5 + 1)
            s_array = trace[sample_S_min:sample_S_max]
            new_trace = np.append(new_trace, s_array)

            ## Get P- & S- windows also from the G matrix:
            G_P = G[sample_P_min:sample_P_max, :]
            G_S = G[sample_S_min:sample_S_max, :]
            G_new = np.vstack((G_P, G_S)) if i == 0 else np.vstack((G_new, G_P, G_S))
        return G_new, new_trace

    def def_phases(self):
        # mqs2019nspq
        a = 1
        phases = []
        phases.append(dict(starttime=lambda dist, depth: origin.time + dist / 2.8,
                           endtime=lambda dist, depth: origin.time + dist / 2.6,
                           comp='Z',
                           fmin=1. / 20.,
                           fmax=1. / 10.,
                           dt=5.0,
                           name='R1_10_20'))
        # phases.append(dict(starttime=lambda dist, depth: origin.time + get_P(depth, dist) - 5,
        #                   endtime=lambda dist, depth: origin.time + get_P(depth, dist) + 35,
        #                   comp='Z',
        #                   fmin=1./20.,
        #                   fmax=1./5.,
        #                   dt=2.0,
        #                   name='P_10_20'))
        phases.append(dict(starttime=lambda dist, depth: origin.time + dist / 2.8,
                           endtime=lambda dist, depth: origin.time + dist / 2.6,
                           comp='Z',
                           fmin=1. / 16.,
                           fmax=1. / 8.,
                           dt=4.0,
                           name='R1_08_16'))
        phases.append(dict(starttime=lambda dist, depth: origin.time + dist / 2.8,  # 2.7,
                           endtime=lambda dist, depth: origin.time + dist / 2.5,  # 2.5,
                           comp='Z',
                           fmin=1. / 32.,
                           fmax=1. / 16.,
                           dt=8.0,
                           name='R1_16_32'))
        phases.append(dict(starttime=lambda dist, depth: origin.time + dist / 2.8,  # /2.7,
                           endtime=lambda dist, depth: origin.time + dist / 2.4,  # /2.5,
                           comp='Z',
                           fmin=1. / 48.,
                           fmax=1. / 24.,
                           dt=12.0,
                           name='R1_24_48'))
        # phases.append(dict(starttime=lambda dist, depth: origin.time+(3389.5*2*np.pi - dist)/2.7,
        #                   endtime=lambda dist, depth: origin.time+(3389.5*2*np.pi - dist)/2.5,
        #                   comp='Z',
        #                   fmin=1./32.,
        #                   fmax=1./16.,
        #                   dt=8.0,
        #                   name='R2_16_32'))
        phases.append(dict(starttime=lambda dist, depth: origin.time + dist / 3.2,  # /3.15,
                           endtime=lambda dist, depth: origin.time + dist / 2.85,  # /2.95,
                           comp='T',
                           fmin=1. / 48.,
                           fmax=1. / 24.,
                           dt=5.0,
                           name='G1_24_48'))
        phases.append(dict(starttime=lambda dist, depth: origin.time + dist / 3.25,  # /3.2,
                           endtime=lambda dist, depth: origin.time + dist / 2.9,  # /2.95,
                           comp='T',
                           fmin=1. / 32.,
                           fmax=1. / 16.,
                           dt=8.0,
                           name='G1_16_32'))
        phases.append(dict(starttime=lambda dist, depth: origin.time + dist / 3.2,  # /3.15,
                           endtime=lambda dist, depth: origin.time + dist / 2.9,  # /2.95,
                           comp='T',
                           fmin=1. / 24.,
                           fmax=1. / 12.,
                           dt=6.0,
                           name='G1_12_24'))
        phases.append(dict(starttime=lambda dist, depth: origin.time + dist / 3.15,  # /3.1,
                           endtime=lambda dist, depth: origin.time + dist / 2.9,  # /3.00,
                           comp='T',
                           fmin=1. / 16.,
                           fmax=1. / 8.,
                           dt=4.0,
                           name='G1_08_16'))
        return phases


class Seismogram:
    def __init__(self, PARAMETERS, db):
        self.par = PARAMETERS
        self.db = db
        self.receiver = instaseis.Receiver(latitude=PARAMETERS['la_r'], longitude=PARAMETERS['lo_r'],
                                           network=PARAMETERS['network'], station=PARAMETERS['station'])
        self.source = instaseis.Source.from_strike_dip_rake(latitude=PARAMETERS['la_s'], longitude=PARAMETERS['lo_s'],
                                                            depth_in_m=PARAMETERS['depth_s'],
                                                            strike=PARAMETERS['strike'], dip=PARAMETERS['dip'],
                                                            rake=PARAMETERS['rake'], M0=PARAMETERS['M0'])

    def get(self):
        traces = self.db.get_seismograms(source=self.source, receiver=self.receiver, components=self.par['components'],
                                         kind=self.par['kind'])
        seismogram = np.array([])
        for trace in traces.traces:
            seismogram = np.append(seismogram, trace)
        return seismogram, traces, self.source


class Green_functions:
    ## Obtain Greens-function from Instaseis - [MxN] : M-number of rows, N-number of columns
    def __init__(self, PARAMETERS, db):
        self.par = PARAMETERS
        self.db = db

    def get(self):
        gf = self.db.get_greens_function(epicentral_distance_in_degree=PARAMETERS['epi'],
                                         source_depth_in_m=PARAMETERS['depth_s'], origin_time=PARAMETERS['origin_time'],
                                         kind=PARAMETERS['kind'], kernelwidth=PARAMETERS['kernelwidth'],
                                         definition=PARAMETERS['definition'])
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
        G[0:G_z, 0] = zss * (0.5) * np.cos(2 * np.deg2rad(PARAMETERS['az'])) - zdd * 0.5
        G[0:G_z, 1] = - zdd * 0.5 - zss * (0.5) * np.cos(2 * np.deg2rad(PARAMETERS['az']))
        # G[0:G_z, 1] =  zdd * (1/3) + zep * (1/3)
        G[0:G_z, 2] = zss * np.sin(2 * np.deg2rad(PARAMETERS['az']))
        G[0:G_z, 3] = -zds * np.cos(np.deg2rad(PARAMETERS['az']))
        G[0:G_z, 4] = -zds * np.sin(np.deg2rad(PARAMETERS['az']))

        G[G_z:G_r, 0] = rss * (0.5) * np.cos(2 * np.deg2rad(PARAMETERS['az'])) - rdd * 0.5
        G[G_z:G_r, 1] = -0.5 * rdd - rss * (0.5) * np.cos(2 * np.deg2rad(PARAMETERS['az']))
        # G[G_z:G_r, 1] =  rdd * (1/3) + rep * (1/3)
        G[G_z:G_r, 2] = rss * np.sin(2 * np.deg2rad(PARAMETERS['az']))
        G[G_z:G_r, 3] = -rds * np.cos(np.deg2rad(PARAMETERS['az']))
        G[G_z:G_r, 4] = -rds * np.sin(np.deg2rad(PARAMETERS['az']))

        G[G_r:G_t, 0] = -tss * (0.5) * np.sin(2 * np.deg2rad(PARAMETERS['az']))
        G[G_r:G_t, 1] = tss * (0.5) * np.sin(2 * np.deg2rad(PARAMETERS['az']))
        # G[G_r:G_t, 1] =   0
        G[G_r:G_t, 2] = tss * np.cos(2 * np.deg2rad(PARAMETERS['az']))
        G[G_r:G_t, 3] = tds * np.sin(2 * np.deg2rad(PARAMETERS['az']))
        G[G_r:G_t, 4] = -tds * np.cos(2 * np.deg2rad(PARAMETERS['az']))
        return G


class Forward_problem:
    def __init__(self, PARAMETERS, G, moment):
        self.par = PARAMETERS
        self.moment = moment
        self.G = G

    def Solve_forward(self):
        # Forward model:
        data = np.matmul(self.G, self.moment)
        return data


class Inversion_problem:
    def __init__(self, data, G, PARAMETERS):
        self.alpha = PARAMETERS['alpha']
        self.beta = PARAMETERS['beta']
        self.m_ref = PARAMETERS['m_ref']
        self.data = data
        self.G = G

    def Solve_LS(self):
        M = np.linalg.lstsq(self.G, self.data)
        print('Least-square: \n %s' % M[0])
        return M

    def Solve_regularization(self):
        M = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.G.T, self.G)), self.G.T), self.data)
        print('Regularization: \n %s' % M)
        return M

    def Solve_damping(self):
        I = np.eye(self.m_ref.__len__())
        M = np.matmul(np.linalg.inv(np.matmul(self.G.T, self.G) + (I * self.alpha ** 2)),
                      (np.matmul(self.G.T, self.data) + np.matmul((I * self.alpha ** 2), self.m_ref)))
        print('Damping : \n%s' % M)
        return M

    def Solve_damping_smoothing(self):
        I = np.eye(self.m_ref.__len__())
        I[1, :] = 0  # Because m1+m2+m3=0
        trace = np.array([1, 1, 0, 0, 0])
        trace.shape = (1, 5)
        trace_matrix = np.matmul(trace.T, trace)
        M = np.matmul(
            np.linalg.inv(np.matmul(self.G.T, self.G) + (I * self.alpha ** 2) + self.beta ** 2 * trace_matrix),
            (np.matmul(self.G.T, self.data) + np.matmul((I * self.alpha ** 2), self.m_ref)))
        print('Damping & smoothening:\n %s' % M)
        return M

    def Solve_SVD(self):
        ## Solve Singular Value Decomposition (SVD)
        U, s, V = np.linalg.svd(self.G, full_matrices=True)
        s_diag = np.zeros((U.__len__(), V.__len__()))
        s_diag[:V.__len__(), :V.__len__()] = np.diag(s)
        M_SVD = np.matmul(np.matmul(V, np.linalg.pinv(s_diag)), np.matmul(U.T, self.data))
        print('SVD: \n%s' % M_SVD)

        # For regularization more work needs to be done:
        # T_diag = np.diag((s ** 2) / (s ** 2 + self.alpha))
        # M_regul = np.matmul(np.matmul(np.matmul(np.matmul(V, T_diag), np.linalg.pinv(s_diag)), U.T), self.data)
        return M_SVD


class Misfit:
    def get_RMS(self, data_obs, data_syn):
        N = data_syn.__len__()  # Normalization factor
        RMS = np.sqrt(np.sum(data_obs - data_syn) ** 2 / N)
        return RMS

    def get_xi(self, data_obs, data_syn, var_est):
        d_obs_mean = np.mean(data_obs)
        var = var_est * d_obs_mean
        likelihood = np.matmul((data_obs - data_syn).T, (data_obs - data_syn)) / (2 * (var ** 2))
        return likelihood

    def norm(self, data_obs, data_syn):
        norm = np.linalg.norm(data_obs - data_syn)
        return norm


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

        # Time sampler: !!!!!!!!!!!!!!!!NOT CORRECT YET!!!!!!!!!!
        year = self.par['origin_time'].year
        month = self.par['origin_time'].month
        day = self.par['origin_time'].day
        hour = int(np.random.uniform(self.sampler['time_range'], self.par['origin_time'].hour))
        print("hour=%i" % hour)
        min = int(np.random.uniform(0, 60))
        print("min=%i" % min)
        sec = int(np.random.uniform(1, 60))
        print("sec=%i" % sec)
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
        d_syn = np.matmul(G, moment)
        return d_syn

    def do(self, number_sample):
        accept = {'epi': np.array([]), 'azimuth': np.array([]), 'depth': np.array([]),
                  'time': {'year': np.array([], dtype='i'), 'month': np.array([], dtype='i'),
                           'day': np.array([], dtype='i'), 'hour': np.array([], dtype='i'),
                           'min': np.array([], dtype='i'), 'sec': np.array([], dtype='i')}, 'misfit': np.array([])}
        ## Starting parameters and create A START MODEL (MODEL_OLD):
        epi_old, azimuth_old, depth_old, time_old = self.model_samples()
        d_syn_old = self.G_function(epi_old, depth_old, azimuth_old, time_old)
        misfit = Misfit()

        Xi_old = misfit.get_xi(self.d_obs, d_syn_old, self.sampler['var_est'])
        accept['epi'] = np.append(accept['epi'], epi_old)
        accept['azimuth'] = np.append(accept['azimuth'], azimuth_old)
        accept['depth'] = np.append(accept['depth'], depth_old)
        accept['time']['year'] = np.append(accept['time']['year'], time_old.year)
        accept['time']['month'] = np.append(accept['time']['month'], time_old.month)
        accept['time']['day'] = np.append(accept['time']['day'], time_old.day)
        accept['time']['hour'] = np.append(accept['time']['hour'], time_old.hour)
        accept['time']['min'] = np.append(accept['time']['min'], time_old.minute)
        accept['time']['sec'] = np.append(accept['time']['sec'], time_old.second)
        accept['misfit'] = np.append(accept['misfit'], Xi_old)

        for i in range(number_sample):
            epi, azimuth, depth, time = self.model_samples()
            d_syn = self.G_function(epi, depth, azimuth, time)
            misfit = Misfit()
            # Xi_norm=misfit.norm(self.d_obs,d_syn)
            Xi_new = misfit.get_xi(self.d_obs, d_syn, self.sampler['var_est'])
            random = np.random.random_sample((1,))
            if Xi_new < Xi_old or (Xi_old / Xi_new) > random:
                accept['epi'] = np.append(accept['epi'], epi)
                accept['azimuth'] = np.append(accept['azimuth'], azimuth)
                accept['depth'] = np.append(accept['depth'], depth)
                accept['time']['year'] = np.append(accept['time']['year'], time.year)
                accept['time']['month'] = np.append(accept['time']['month'], time.month)
                accept['time']['day'] = np.append(accept['time']['day'], time.day)
                accept['time']['hour'] = np.append(accept['time']['hour'], time.hour)
                accept['time']['min'] = np.append(accept['time']['min'], time.minute)
                accept['time']['sec'] = np.append(accept['time']['sec'], time.second)
                accept['misfit'] = np.append(accept['misfit'], Xi_new)
                Xi_old = Xi_new
            else:
                continue
        filepath = self.par['filepath']
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


    def make_PDF(self):
        if os.path.isfile(filepath) == True:
            with open(self.par['filepath'], 'r') as stream:
                data_loaded = yaml.load(stream)
                stream.close()
            # NOT FINISHED YET
        else:
            print("The filename does not exist yet, the MH algorithm will now first run")
            self.do(self.sampler['sample_number'])
            print("The file is now available, so run make_PDF again")


class grid_search_algorithm:
    def __init__(self, PARAMETERS, sampler, db, data):
        self.db = db
        self.par = PARAMETERS
        self.sampler = sampler
        self.d_obs = data

        epi = np.linspace(self.sampler['epi']['range_min'], self.sampler['epi']['range_max'],
                          self.sampler['epi']['steps'])
        azimuth = np.linspace(self.sampler['azimuth']['range_min'], self.sampler['azimuth']['range_max'],
                              self.sampler['azimuth']['steps'])
        depth = np.linspace(self.sampler['depth']['range_min'], self.sampler['depth']['range_max'],
                            self.sampler['depth']['steps'])

        # Time sampler: !!!!!!!!!!!!!!!!NOT CORRECT YET!!!!!!!!!!
        # time_obs = self.par['origin_time'] # Observed time at station
        # time_sample_min = obspy.UTCDateTime(time_obs.year, time_obs.month, time_obs.day,self.sampler['time_range'] , time_obs.min, time_obs.sec)
        # time = np.linspace(time_sample_min,time_obs,step)

        self.dict = {'epi': epi,
                     'azimuth': azimuth,
                     'depth': depth}

    def make_grid(self):
        # Call the keys within the dictionary:
        self.keys = self.dict.keys()

        # Call the values within the dictionary:
        self.val = self.dict.values()

        # NOT FINISHED YET


class Plots:
    def __init__(self):
        params = {'legend.fontsize': 'x-large',
                  'figure.figsize': (10, 10),
                  'axes.labelsize': 25,
                  'axes.titlesize': 'x-large',
                  'xtick.labelsize': 25,
                  'ytick.labelsize': 25}
        pylab.rcParams.update(params)

    def Log_G(self, G):
        plt.imshow(np.log(G))
        # plt.title('Matrix : G')
        plt.gca().set_aspect(1.0 / 10000.0)
        # plt.gca().set_aspect(10000.0)
        plt.show()

    def G_transpose_G(self, G):
        plt.imshow(np.matmul(G.T, G))
        # plt.title('G.T G')
        plt.show()

    def Beachball(self, moment):
        beachball(moment, size=200, linewidth=2, facecolor='b')

    def Compare_seismograms(self, forward_data, instaseis_data):
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(121)
        ax1.plot(forward_data)
        plt.title('Seismogram calculated with forward model')
        plt.xlabel('t')
        plt.ylabel('Forward data displacement [m]')
        plt.grid()

        ax2 = fig.add_subplot(122)
        ax2.plot(instaseis_data)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        plt.title('Seismogram calculated with Instaseis')
        plt.xlabel('t')
        plt.ylabel('Instaseis data displacement [m]')
        plt.grid()
        plt.show()


if __name__ == '__main__':
    main()
