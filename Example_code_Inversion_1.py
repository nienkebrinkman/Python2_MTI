## Inversion MT : Inverting for Epicentral distance and Depth [Indirectly also for m_xx,m_yy,m_xy,m_xz,m_yz

# IMPORTANT: in Get_Parameters --> sdr = False !!!!!!!

import instaseis
import numpy as np
import matplotlib.pylab as plt


## All different classes:
from Create_observed import Create_observed
from Get_Parameters import Get_Paramters
from Green_functions import Green_functions
from Inversion_problems import Inversion_problem
from Forward_problem import Forward_problem
from Misfit import Misfit

# Getting parameters to create the observed data:
get_parameters = Get_Paramters()
PARAMETERS = get_parameters.get_unkown()  # These values can only be used to create d_obs
PRIOR = get_parameters.get_prior(PARAMETERS['epi'])
VALUES = get_parameters.specifications()
PRIOR['az'] = PARAMETERS['az']
PRIOR['baz'] = PARAMETERS['baz']

# Initiate the databases from instaseis:
db = instaseis.open_db(PRIOR['VELOC'])

## Step 1 - Create the observed data:
create = Create_observed(PRIOR, PARAMETERS, db)
# FUll seismogram:
# d_obs = stacked numpy array of the seismogram
# traces = obspy stream with 3 traces [Z,R,T]
# source = instaseis source
d_obs, traces, source = create.get_seis_automatic(prior=PRIOR, noise_model=VALUES['noise'], sdr=VALUES['sdr'])

# P and S windows of the seismogram:
# traces_obs = obspy stream with 3 traces [Z,R,T], containing both P and S windows
# p_obs      = obspy stream with 3 traces [Z,R,T], containing ONLY P windows
# s_obs      = obspy stream with 3 traces [Z,R,T], containing ONLY S windows
traces_obs, p_obs, s_obs = create.get_window_obspy(traces, PARAMETERS['epi'], PARAMETERS['depth_s'],
                                                   PARAMETERS['origin_time'], VALUES['npts'])
v_stack_traces_obs = 1

# The relative origin time that is going to be used during further process, since origin time is supposed to be unknown:
# time_at_receiver = UTCDateTime
time_at_receiver = create.get_receiver_time(PARAMETERS['epi'], PARAMETERS['depth_s'], traces)

# The amount of seconds between the first arriving P and S waves:
# time_between_windows = float64
time_between_windows = create.time_between_windows(PARAMETERS['epi'], PARAMETERS['depth_s'], traces[0].meta.delta)

moment_init = np.array([source.m_tt, source.m_pp, source.m_tp, source.m_rt, source.m_rp])
print("Initial moment: \nm_xx = %.1f \nm_yy=%.1f \nm_xy=%.1f \nm_xz=%.1f\nm_yz=%.1f" % (
source.m_tt, source.m_pp, -source.m_tp, source.m_rt, -source.m_rp))

## Step 2 - Parameters = epi,depth
# For now the epi,depth,time are using the exact same values as the prior, because I wanted to test my code
epi = PARAMETERS['epi']
depth = PARAMETERS['depth_s']

## Step 3 - Generate G-functions with these parameters
# G_tot = v_stack of G_z,G_r,G_t
green = Green_functions(PRIOR, db)
G_tot, G_z, G_r, G_t = green.get(time_at_receiver, epi, depth, VALUES['npts'])

## Step 4 - Calculate moment
inv = Inversion_problem(PRIOR)
moment = inv.Solve_damping_smoothing(d_obs, G_tot)

## Step 5 - Calculate synthetic data
forward = Forward_problem(PARAMETERS, G_tot, moment)
d_syn = forward.Solve_forward()

plt.plot(d_syn)
plt.plot(d_obs, linestyle=':')
plt.show()

## Step 6 - Calculate the misfit between d_syn and d_obs
misfit = Misfit(VALUES['directory'])
# TODO - split the d_syn into an obspy stream for 3 components to calculate the misfit
