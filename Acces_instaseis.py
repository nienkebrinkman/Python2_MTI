import instaseis
import obspy
import numpy as np
import os.path
from obspy.geodetics import kilometer2degrees
from obspy.geodetics.base import gps2dist_azimuth

## All different classes:
from Metropolis_Hasting import MH_algorithm
from Inversion_problems import Inversion_problem
from Forward_problem import Forward_problem
from Seismogram import Seismogram
from Green_functions import Green_functions
from Source_code import Source_code
from Plots import Plots
from MCMC_pymc import MCMC_algorithm
from Neighborhood_algorithm import Neighborhood_algorithm
from Misfit import Misfit

## Velocity model:
VELOC = 'http://instaseis.ethz.ch/marssynthetics/C30VL-AKSNL-1s'
VELOC_taup = 'iasp91'

# Safe parameters:
directory = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/testdata'
filename = '100000_trial.yaml'
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
    'VELOC_taup':VELOC_taup}

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
    'var_est': 'var_est',
    'directory': directory,
    'filename': filename,
    'filepath': filepath }
sampler['depth']['range_min'] = 800
sampler['depth']['range_max'] = 1200
sampler['depth']['step'] = 20
sampler['epi']['range_min'] = 20
sampler['epi']['range_max'] = 40
sampler['epi']['step'] = 40
sampler['azimuth']['range_min'] = 2
sampler['azimuth']['range_max'] = 22
sampler['azimuth']['step'] = 22
sampler['time_range'] = PARAMETERS['origin_time'].hour - 1  # The time range can only vary may be two hours!!
sampler['sample_number'] = 5
sampler['var_est'] = 0.05


def main():

    ## Plot marginal 2d Histogram of Data from Metropolis hasting
    # plot=Plots()
    # plot.make_PDF(sampler)

    ## Obtain database to create both Seismograms and Greens_functions:

    db = instaseis.open_db(VELOC)

    ## Make seismogram:

    seis = Seismogram(PARAMETERS, db)
    u, traces, source = seis.get()  # u = stacked seismograms , traces = 3 component seismogram separated

    ## MCMC algorithm

    MCMC = MCMC_algorithm(sampler,u)
    MCMC.Instaseis(PARAMETERS, db)


    ## Metropolis Hasting Algorithm

    MH = MH_algorithm(PARAMETERS, sampler, db, u)
    # Parallel = MH.do_parallel()
    accept_model = MH.do()

    ## Get Green functions:
    Green = Green_functions(PARAMETERS, db)
    G = Green.get()

    ## Obtain Seismogram and Green function with certain window
    source_inv = Source_code(PARAMETERS, db)
    G_window, u_window = source_inv.get_windows(traces, G)

    ## Solve forward model:
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

    # Plot observed vs synthetic seismogram:
    plot=Plots()
    plot.Compare_seismograms(data,u)


if __name__ == '__main__':
    main()
