## Inversion 1 : Only inverting for Depth/Epicentral_distance/Origin_time

import instaseis
import numpy as np
import matplotlib.pylab as plt

## All different classes:
from Get_Parameters import Get_Paramters
from Metropolis_Hasting import MH_algorithm
from Inversion_problems import Inversion_problem
from Forward_problem import Forward_problem
from Seismogram import Seismogram
from Source_code import Source_code
from Plots import Plots
from Misfit import Misfit
from test import MH_algorithm_test


# Getting parameters to create the observed data:
get_parameters = Get_Paramters()
PARAMETERS = get_parameters.get_initial_par()
sampler = get_parameters.get_MHMC_par()

# Print the moment that you start with (This is normally unknown and you want to get as close as possible to this one)
moment_given = np.array(
    [PARAMETERS['m_tt'], PARAMETERS['m_pp'], PARAMETERS['m_tp'], PARAMETERS['m_rt'], PARAMETERS['m_rp']])
print(moment_given)

# Initiate the databases from instaseis:
db = instaseis.open_db(PARAMETERS['VELOC'])

## Step 1 - Calculate an observed seismogram = d_obs -- With or Without noise
seis = Seismogram(PARAMETERS, db)
# Without noise:
d_obs, traces, source = seis.get_seis_automatic(sdr=False)
# With noise:
# d_obs, traces, source = seis.get_seis_automatic_with_noise(noise_model=PARAMETERS['noise_model'],sdr=False)


# print(source) - This is the moment resulting from instaseis (should be the same as moment_given)
moment_init = np.array([source.m_tt, source.m_pp, source.m_tp, source.m_rt, source.m_rp])
print(moment_init)

## Step 2 - Parameters = epi,depth,time
epi = PARAMETERS['epi']
depth = PARAMETERS['depth_s']
time =PARAMETERS['origin_time']
# For now the epi,depth,time are using the exact same values as the prior, because I wanted to test my code

## Step 3 - Generate G-functions with these parameters
testsample = MH_algorithm(PARAMETERS, sampler, db, d_obs, traces)
G = testsample.generate_G(epi,depth,time)

## Obtain Seismogram and Green function with certain window
window = Source_code(PARAMETERS['VELOC_taup'])
G_window = window.get_G(traces,G,PARAMETERS['epi'],PARAMETERS['depth_s'])


## Step 4 - Calculate moment
inv = Inversion_problem(PARAMETERS)
moment = inv.Solve_damping_smoothing(d_obs, G_window)
moment_transfer = np.array([moment[0], moment[1], -moment[2], moment[3], -moment[4]])  # Transferring your moment!!!


## Step 5 - Calculate synthetic data
forward = Forward_problem(PARAMETERS,G_window,moment)
d_syn = forward.Solve_forward()

## Step 6 - Calculate the misfit between d_syn and d_obs
mis = Misfit()
dt = 1 / (traces[0].meta.sampling_rate)
misfit = mis.get_xi(d_obs,d_syn,sampler['var_est'],dt)
D,time_shift = mis.get_CC(d_obs,d_syn,dt)
