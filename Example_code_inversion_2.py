## Inversion 2 : Inverting for Depth/Epicentral_distance/Stike/Dip/Slip

import instaseis
import numpy as np
import matplotlib.pylab as plt
import geographiclib.geodesic as geo
import pylab

##
from Create_observed import Create_observed
from Get_Parameters import Get_Paramters
from Seismogram import Seismogram
from Source_code import Source_code
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

## Step 3 - Parameters = Strike,dip,rake
strike= PARAMETERS['strike']
dip = PARAMETERS['dip']
rake = PARAMETERS['rake']

## Step 4 - Calculate d_syn with the use of get_seismogram (from instaseis)
seis = Seismogram(PRIOR,db)
window_code = Source_code(PRIOR['VELOC_taup'])
dict = geo.Geodesic(a=PRIOR['radius'], f=0).ArcDirect(lat1=PRIOR['la_r'], lon1=PRIOR['lo_r'],
                                                      azi1=PRIOR['baz'],
                                                      a12=epi, outmask=1929)

# FULL seismogram
# d_obs = stacked numpy array of the seismogram
# traces = obspy stream with 3 traces [Z,R,T]
# source = instaseis source
d_syn, tr_syn, sources = seis.get_seis_manual(la_s=dict['lat2'], lo_s=dict['lon2'], depth=depth,
                                              strike=79, dip=50, rake=20,
                                              time=time_at_receiver, sdr=True)


# traces_syn = obspy stream with 3 traces [Z,R,T], containing both P and S windows
# p_syn      = obspy stream with 3 traces [Z,R,T], containing ONLY P windows
# s_syn      = obspy stream with 3 traces [Z,R,T], containing ONLY S windows
traces_syn, p_syn, s_syn = window_code.get_window_obspy(tr_syn,epi, depth, time_at_receiver, VALUES['npts'])

## Step 5- Calculate the misfit between d_obs and d_syn
# xi_L2 = misfit using L2
# xi_CC = misfit using Cross Correlation
# shifts_L2 = shifts for the best cross correlation fit, DONE for all 5 phases: Z: P&S, R: P&S, T: S
# shifts_CC = shifts for the best cross correlation fit, DONE for all 5 phases: Z: P&S, R: P&S, T: S


misfit = Misfit(VALUES['directory'])
xi_L2, shifts_L2 = misfit.L2_stream(p_obs, p_syn, s_obs, s_syn, time_at_receiver, PRIOR['var_est'])
xi_CC, shifts_CC = misfit.CC_stream(p_obs, p_syn, s_obs, s_syn, time_at_receiver)
print("L2 misfit : %.1f" %xi_L2)
print("CC misfit : %.1f" %xi_CC)


## Step 6 - Plot
# total_stream --> usefull for the following plot!
total_stream = traces_obs.__add__(traces_syn)

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (20, 15),
          'axes.labelsize': 25,
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 25,
          'ytick.labelsize': 25}
pylab.rcParams.update(params)
f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)

ax1.plot(total_stream[0], label="Z_obs")
ax2.plot(total_stream[1], label="R_obs")
ax3.plot(total_stream[2], label="T_obs")
ax1.plot(total_stream[3], label="Z_syn", linestyle=':')
ax2.plot(total_stream[4], label="R_syn", linestyle=':')
ax3.plot(total_stream[5], label="T_syn", linestyle=':')
ax1.legend()
ax2.legend()
ax3.legend()
plt.xlabel('time')
plt.show()
plt.close()

