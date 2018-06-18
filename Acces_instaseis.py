import instaseis
import geographiclib.geodesic as geo
import pylab
import matplotlib.pylab as plt
from obspy.core.trace import Trace
import obspy.signal.cross_correlation as cc
import numpy as np

## All different classes:
from Get_Parameters import Get_Paramters
from Source_code import Source_code
from MCMC_stream import MCMC_stream
from Seismogram import Seismogram
from Misfit import Misfit
from Create_observed import Create_observed
from Green_functions import Green_functions
from Inversion_problems import Inversion_problem

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
create = Create_observed(PRIOR,PARAMETERS,db)

d_obs, traces, source = create.get_seis_automatic(prior=PRIOR,noise_model=VALUES['noise'],sdr=VALUES['sdr'])
traces.detrend(type='demean')
traces_obs, p_obs, s_obs = create.get_window_obspy(traces,PARAMETERS['epi'],PARAMETERS['depth_s'],PARAMETERS['origin_time'],VALUES['npts'])


# traces_obs.detrend(type='demean')
# traces_obs.plot()

# for tr in traces_obs:
#     tr.detrend()
# traces_obs.plot()
# time_at_receiver = create.get_receiver_time(PARAMETERS['epi'],PARAMETERS['depth_s'],traces)
time_between_windows = create.time_between_windows(PARAMETERS['epi'],PARAMETERS['depth_s'],traces[0].stats.delta)
time_at_receiver = PARAMETERS['origin_time']
# traces_obs.plot()
# traces_obs.detrend(type='demean')
# traces_obs.plot()
# traces_obs.filter("highpass",freq = 1.0/90.0)
# traces_obs.plot()

#----------------------------------------------------------------------------------------------------------------------#

# Now we can Run a Monte Carlo algorthm:
M_algorithm = MCMC_stream(traces_obs,p_obs,s_obs, PRIOR,db,VALUES,time_at_receiver)
# M_algorithm.run_parallel()
M_algorithm.start_MCMC(savepath=VALUES['directory'] + '/mcmc_stream.txt')
#
#
# seis = Seismogram(PRIOR,db)
# window_code = Source_code(PRIOR['VELOC_taup'])
#
#
# dep = np.linspace(PRIOR['depth']['range_min'], PRIOR['depth']['range_max'],100)
# # # dep = np.ones(100) * 10000
# epi = np.linspace(PRIOR['epi']['range_min'],PRIOR['epi']['range_max'],100)
# #
# dep = [PARAMETERS['depth_s'],43109.100,42987.3]
# epi = [PARAMETERS['epi'],45.9912, 46.0613]
#
# total_CC = np.array([])
# total_L2 = np.array([])
# total = np.array([])
# accepted = 0
# rejected =0
# file_path = VALUES['directory'] + '/mcmc_epi_dep.txt'
# with open(file_path, 'w') as save_file:
#     for i in range(10000):
#         # for j in range(100):
#         if i == 0:
#             epi,dep = M_algorithm.model_samples()
#         else:
#             epi,dep = M_algorithm.model_samples(epi_old,dep_old)
#             if epi < PRIOR['epi']['range_min'] or epi > PRIOR['epi']['range_max'] or dep < \
#                     PRIOR['depth']['range_min'] or dep > PRIOR['depth']['range_max']:
#                 continue
#
#
#         dict = geo.Geodesic(a=PRIOR['radius'], f=0).ArcDirect(lat1=PRIOR['la_r'], lon1=PRIOR['lo_r'],
#                                                      azi1=PRIOR['baz'],
#                                                      a12=epi, outmask=1929)
#         d_syn, tr_syn, sources = seis.get_seis_manual(la_s=dict['lat2'], lo_s=dict['lon2'], depth=dep,
#                                                            strike=79, dip=50, rake=20,
#                                                            time=time_at_receiver, sdr=True)
#
#
#         traces_syn, p_syn, s_syn = window_code.get_window_obspy(tr_syn,epi,dep,time_at_receiver,VALUES['npts'])
#
#         # total_stream = traces_obs.__add__(traces_syn)
#         #
#         # #
#         # # params = {'legend.fontsize': 'x-large',
#         # #           'figure.figsize': (20, 15),
#         # #           'axes.labelsize': 25,
#         # #           'axes.titlesize': 'x-large',
#         # #           'xtick.labelsize': 25,
#         # #           'ytick.labelsize': 25}
#         # # pylab.rcParams.update(params)
#         # f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
#         #
#         #
#         # ax1.plot(total_stream[0],label = "Z_obs")
#         # ax2.plot(total_stream[1],label = "R_obs")
#         # ax3.plot(total_stream[2],label = "T_obs")
#         # ax1.plot(total_stream[3],label = "Z_syn",linestyle=':')
#         # ax2.plot(total_stream[4],label = "R_syn",linestyle=':')
#         # ax3.plot(total_stream[5],label = "T_syn",linestyle=':')
#         # ax1.legend()
#         # ax2.legend()
#         # ax3.legend()
#         # plt.xlabel('time')
#         # # plt.show()
#         # plt.savefig(VALUES['directory'] + '/Seismogram_plots'+ '/_%i_%i' %(i,j) )
#         # plt.close()
#
#         misfit = Misfit(VALUES['directory'])
#         # xi_L2_new, shifts_L2 = misfit.L2_stream(p_obs,p_syn,s_obs,s_syn,time_at_receiver,PRIOR['var_est'])
#         # total_L2 = np.append(total_L2, xi_L2_new)
#
#         xi_CC_new, shifts_CC = misfit.CC_stream( p_obs, p_syn, s_obs, s_syn,time_at_receiver)
#         total_CC = np.append(total_CC,xi_CC_new)
#         # print(epi[j])
#         # print(dep[i])
#         # print(misfit)
#         # save_file.write("%.4f,%.4f,%.4f,%i,%i\n\r" % (epi[j], dep[i],xi_CC_new,i,j))
#
#         if i == 0:
#             xi_CC_old = xi_CC_new
#             epi_old = epi
#             dep_old = dep
#             save_file.write("%.4f,%.4f,%.4f,%i\n\r" % (epi, dep, 0.00000,i))
#             continue
#         else:
#
#             random = np.random.random_sample((1,))
#             if xi_CC_new < xi_CC_old or np.exp((xi_CC_old - xi_CC_new) / VALUES['temperature']) > random:
#
#                 accepted +=1
#                 print("accept = %i" % accepted)
#                 save_file.write("%.4f,%.4f,%.4f,%i\n\r" % (epi, dep, xi_CC_new,i))
#
#                 shift, cc_max = cc.xcorr_3c(traces_obs, traces_syn, s_obs[0].meta.npts, components=['Z', 'R', 'T'], full_xcorr=False,
#                                             abs_max=True)
#                 total = np.append(total,cc_max)
#                 xi_CC_old = xi_CC_new
#                 epi_old = epi
#                 dep_old = dep
#             else:
#                 rejected += 1
#                 print("reject = %i" % rejected)
#
# save_file.close()
#
# #
#
