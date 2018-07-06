# This code will run the MCMC algorithm

# IMPORTANT: in Get_Parameters --> MCMC = 'M' or MCMC = 'MH' (see Get_Parameters for further explanation)
import instaseis
import numpy as np
import geographiclib.geodesic as geo
import pylab
import matplotlib.pylab as plt
from obspy.core.trace import Trace
import obspy.signal.cross_correlation as cc

## All different classes:
from Get_Parameters import Get_Paramters
from Surface_waves import Surface_waves
from Create_observed import Create_observed
from MCMC_stream import MCMC_stream
from Seismogram import Seismogram
from Source_code import Source_code
from Misfit import Misfit


# Initiate Parameters:
get_parameters = Get_Paramters()
PARAMETERS = get_parameters.get_unkown()
PRIOR = get_parameters.get_prior()
VALUES = get_parameters.specifications()

## DISCUSS THIS!!!!
PRIOR['az'] = PARAMETERS['az']
PRIOR['baz'] = PARAMETERS['baz']

# Initiate the databases from instaseis:
db = instaseis.open_db(PRIOR['VELOC'])
create = Create_observed(PRIOR, PARAMETERS, db)

d_obs, tr_obs, source = create.get_seis_automatic(prior=PRIOR, noise_model=VALUES['noise'], sdr=VALUES['sdr'])
traces_obs, p_obs, s_obs = create.get_window_obspy(tr_obs, PARAMETERS['epi'], PARAMETERS['depth_s'],
                                                   PARAMETERS['origin_time'], VALUES['npts'])
time_at_receiver = create.get_receiver_time(PARAMETERS['epi'], PARAMETERS['depth_s'], tr_obs)

# create.get_fft(traces=traces, directory=VALUES['directory'])

sw = Surface_waves(PRIOR)
R_env_obs = sw.rayleigh_pick(tr_obs.traces[0], PARAMETERS['la_s'], PARAMETERS['lo_s'], PARAMETERS['depth_s'],
                             VALUES['directory'], PARAMETERS['origin_time'], VALUES['npts'],plot_modus=True)
L_env_obs = sw.love_pick(tr_obs.traces[2], PARAMETERS['la_s'], PARAMETERS['lo_s'], PARAMETERS['depth_s'],
                         VALUES['directory'], PARAMETERS['origin_time'], VALUES['npts'])


# ------------------------------------------------------------------

seis = Seismogram(PRIOR,db)
window_code = Source_code(PRIOR['VELOC_taup'])
misfit = Misfit(VALUES['directory'])

start_sample_path = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/Final/var_50_resume.txt'
# start_sample_path = None

m = MCMC_stream(R_env_obs,L_env_obs,traces_obs,p_obs,s_obs,PRIOR,db,VALUES,time_at_receiver,start_sample_path)
m.start_MCMC(VALUES['directory'] + '/var_50_resumed.txt')

# file_path = VALUES['directory'] + '/MCMC.txt'
# with open(file_path, 'w') as save_file:
#     for i in range(10000):
#         # for j in range(100):
#         if i == 0:
#             epi,depth = m.model_samples()
#             strike, dip, rake = m.model_samples_sdr()
#         else:
#             epi,depth = m.model_samples(epi_old,dep_old)
#             if epi < PRIOR['epi']['range_min'] or epi > PRIOR['epi']['range_max'] or depth < \
#                     PRIOR['depth']['range_min'] or depth > PRIOR['depth']['range_max']:
#                 continue
#             strike, dip, rake = m.model_samples_sdr(strike_old, dip_old, rake_old)
#         dict = geo.Geodesic(a=PRIOR['radius'], f=0).ArcDirect(lat1=PRIOR['la_r'], lon1=PRIOR['lo_r'],
#                                                               azi1=PRIOR['baz'],
#                                                               a12=epi, outmask=1929)
#         d_syn, tr_syn, sources = seis.get_seis_manual(la_s=dict['lat2'], lo_s=dict['lon2'],
#                                                       depth=depth,
#                                                       strike=strike, dip=dip, rake=rake,
#                                                       time=time_at_receiver, sdr=True)
#         L_env_syn = sw.love_pick(tr_syn.traces[2], dict['lat2'], dict['lon2'],depth,
#                                  VALUES['directory'], time_at_receiver, VALUES['npts'])
#         R_env_syn = sw.rayleigh_pick(tr_syn.traces[0], dict['lat2'], dict['lon2'],depth,
#                                      VALUES['directory'], time_at_receiver, VALUES['npts'])
#
#         traces_syn, p_syn, s_syn = window_code.get_window_obspy(tr_syn, epi,depth,
#                                                                 time_at_receiver, VALUES['npts'])
#         misfit = Misfit(VALUES['directory'])
#         xi_CC, shifts_CC = misfit.CC_stream(p_obs, p_syn, s_obs, s_syn, time_at_receiver)
#         xi_R_l2 = misfit.SW_L2(R_env_obs, R_env_syn, PRIOR['var_est'])
#         xi_L_l2 = misfit.SW_L2(L_env_obs, L_env_syn, PRIOR['var_est'])
#
#         total_new = xi_CC + xi_R_l2 + xi_L_l2
#
#         if i == 0:
#             accepted = 0
#             rejected = 0
#             total_old = total_new
#             epi_old = epi
#             dep_old = depth
#             strike_old = strike
#             dip_old = dip
#             rake_old = rake
#             save_file.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n\r" % (epi, depth,strike,dip,rake,total_new,xi_CC,xi_R_l2,xi_L_l2 ))
#             continue
#         else:
#
#             random = np.random.random_sample((1,))
#             if total_new < total_old or np.exp((total_old - total_new) / VALUES['temperature']) > random:
#
#                 accepted +=1
#                 print("accept = %i" % accepted)
#                 save_file.write("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n\r" % (
#                 epi, depth, strike, dip, rake, total_new, xi_CC, xi_R_l2, xi_L_l2))
#
#                 total_old = total_new
#                 epi_old = epi
#                 dep_old = depth
#                 strike_old = strike
#                 dip_old = dip
#                 rake_old = rake
#             else:
#                 rejected += 1
#                 print("reject = %i" % rejected)
#
# save_file.close()
