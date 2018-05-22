## Inversion 2 : Inverting for Depth/Epicentral_distance/Origin_time/Stike/Dip/Slip

import instaseis
import numpy as np
import matplotlib.pylab as plt
import geographiclib.geodesic as geo

## All different classes:
from Get_Parameters import Get_Paramters
from Seismogram import Seismogram
from Source_code import Source_code
from Misfit import Misfit
from Plots import Plots

# Getting parameters to create the observed data:
get_parameters = Get_Paramters()
PARAMETERS = get_parameters.get_initial_par()
sampler = get_parameters.get_MHMC_par()

# Initiate the databases from instaseis:
db = instaseis.open_db(PARAMETERS['VELOC'])

## Step 1 - Calculate an observed seismogram = d_obs -- With or Without noise
seis = Seismogram(PARAMETERS, db)
# Without noise:
d_obs, traces, source = seis.get_seis_automatic(sdr=True)
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

## Step 3 - Calculate Latitude and Longitude for the source from the proposed Parameters:
# dict = geo.Geodesic(a=3389.5, f=0).Direct(lat1=PARAMETERS['la_r'], lon1=PARAMETERS['lo_r'], azi1=PARAMETERS['az'],
#                                                       s12=epi, outmask=1929)
dict = geo.Geodesic(a=3389.5, f=0).ArcDirect(lat1=PARAMETERS['la_r'], lon1=PARAMETERS['lo_r'], azi1=PARAMETERS['baz'],
                                                      a12=epi, outmask=1929)
la_s=dict['lat2']
lo_s=dict['lon2']

## Step 4 - Parameters = Strike,dip,slip
strike= PARAMETERS['strike']
dip = PARAMETERS['dip']
rake = PARAMETERS['rake']

## Step 5 - Calculate d_syn with the use of get_seismogram (from instaseis)
seismogram = Seismogram(PARAMETERS,db)
d_syn, traces_syn, source_syn = seismogram.get_seis_manual(la_s,lo_s,depth,strike,dip,rake,time,sdr=True)
window = Source_code(PARAMETERS['VELOC_taup'])
d_syn_window = window.get_windows(traces_syn, epi, depth)


## Step 6 - Calculate the misfit between d_obs and d_syn
mis = Misfit()
dt = 1 / (traces[0].meta.sampling_rate)
misfit = mis.get_xi(d_obs,d_syn_window,sampler['var_est'],dt)
Xi,time_shift = mis.get_CC(d_obs,d_syn_window,dt)


plt.plot(d_syn_window)
plt.plot(d_obs, linestyle = ':')
plt.show()
a=1

