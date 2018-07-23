# This code will run the MCMC algorithm

# IMPORTANT: in Get_Parameters --> MCMC = 'M' or MCMC = 'MH' (see Get_Parameters for further explanation)
import instaseis
import obspy
import numpy as np
from obspy import read
import geographiclib.geodesic as geo
import pylab
import matplotlib.pylab as plt
from obspy.core.trace import Trace
import obspy.signal.cross_correlation as cc
from obspy.geodetics.base import gps2dist_azimuth
from obspy.geodetics import kilometer2degrees
from mqscatalog import get_phase_picks

## All different classes:
from Get_Parameters import Get_Paramters
from Surface_waves import Surface_waves
from Create_observed import Create_observed
from MCMC_stream import MCMC_stream
from Seismogram import Seismogram
from Source_code import Source_code
from Misfit import Misfit
from Blindtest import Blindtest
from create_starting_sample import create_starting_sample


def main():
    Acces_Blindtest()


def Acces_Normal():
    # Initiate Parameters:

    get_parameters = Get_Paramters()
    PARAMETERS = get_parameters.get_unkown()
    PRIOR = get_parameters.get_prior()
    VALUES = get_parameters.specifications()

    VALUES['directory'] = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/Final'

    ## DISCUSS THIS!!!!
    PRIOR['az'] = PARAMETERS['az']
    PRIOR['baz'] = PARAMETERS['baz']

    # Initiate the databases from instaseis:
    db = instaseis.open_db(PRIOR['VELOC'])
    create = Create_observed(PRIOR, db)

    d_obs, tr_obs, source = create.get_seis_automatic(parameters=PARAMETERS, prior=PRIOR, noise_model=VALUES['noise'],
                                                      sdr=VALUES['sdr'])
    time_at_receiver = create.get_receiver_time(PARAMETERS['epi'], PARAMETERS['depth_s'], PARAMETERS['origin_time'])
    traces_obs, p_obs, s_obs,start_time_p,start_time_s = create.get_window_obspy(tr_obs, PARAMETERS['epi'], PARAMETERS['depth_s'],
                                                       PARAMETERS['origin_time'], VALUES['npts'])


    PRIOR['var_est'] = create.get_var_data(start_time_p,tr_obs)
    # PRIOR['var_est'] =1

    sw = Surface_waves(PRIOR)
    R_env_obs = sw.rayleigh_pick(tr_obs.traces[0], PARAMETERS['la_s'], PARAMETERS['lo_s'], PARAMETERS['depth_s'],
                                 VALUES['directory'], PARAMETERS['origin_time'], VALUES['npts'], plot_modus=False)
    L_env_obs = sw.love_pick(tr_obs.traces[2], PARAMETERS['la_s'], PARAMETERS['lo_s'], PARAMETERS['depth_s'],
                             VALUES['directory'], PARAMETERS['origin_time'], VALUES['npts'], plot_modus=False)
    # tr_obs.plot()
    # ------------------------------------------------------------------

    seis = Seismogram(PRIOR, db)
    window_code = Source_code(PRIOR['VELOC_taup'])
    misfit = Misfit(VALUES['directory'])

    # start_sample_path = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/Final/close_sample.txt'
    start_sample_path = None

    m = MCMC_stream(R_env_obs, L_env_obs, traces_obs, p_obs, s_obs, PRIOR, db, VALUES,PARAMETERS['origin_time'], start_time_p, start_time_s,
                    start_sample_path,None,
                 full_obs_trace=tr_obs)
    m.start_MCMC(VALUES['directory'] + '/Exploring.txt')


def Acces_Blindtest():
    # BLINDTEST_MSEED = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/Database/data_Nienke/M3.5_8213363_deg_2019-02-15.mseed'
    BLINDTEST_MSEED = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/Database/data_Nienke/M5.0_3914855_deg_2019-09-22.mseed'
    BLINDTEST_XML = BLINDTEST_MSEED.replace(".mseed", ".xml")

    # Initiate Parameters:
    get_parameters = Get_Paramters()
    PRIOR = get_parameters.get_prior()
    VALUES = get_parameters.specifications()
    VALUES['npts'] = 30000
    VALUES['directory'] = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/Blindtest'

    # st = read(VALUES['directory'] + '/bw.mseed')
    # st_reject = read(VALUES['directory'] + '/bw_reject.mseed')

    # Initiate the databases from instaseis:
    db = instaseis.open_db(PRIOR['VELOC'])
    tr_obs = obspy.read(BLINDTEST_MSEED)
    # tr_obs.plot(outfile=VALUES['directory'] + '/Observed')
    tr_obs.integrate()
    # tr_obs.plot(outfile=VALUES['directory'] + '/Observed_integrated')
    # source = instaseis.Source.parse(BLINDTEST_XML)
    blindtest = Blindtest()
    events = blindtest.get_events(BLINDTEST_XML)
    # get_parameters.get_prior_blindtest(events[0])
    time, depth, la_s, lo_s = blindtest.get_pref_origin(events[0])

    dist, az, baz = gps2dist_azimuth(lat1=la_s,
                                     lon1=lo_s,
                                     lat2=PRIOR['la_r'],
                                     lon2=PRIOR['lo_r'], a=PRIOR['radius'], f=0)
    epi = kilometer2degrees(dist, radius=PRIOR['radius'])
    PRIOR['az'] = az
    PRIOR['baz'] = baz
    PRIOR['epi']['range_min'] = epi - 5
    PRIOR['epi']['range_max'] = epi + 5
    PRIOR['epi']['spread'] = 1
    PRIOR['depth']['range_min'] = depth - 10000
    PRIOR['depth']['range_max'] = depth + 10000
    PRIOR['network'] = tr_obs.traces[0].meta.network
    PRIOR['location'] = tr_obs.traces[0].meta.location
    PRIOR['station'] = tr_obs.traces[0].meta.station
    est_noise = Create_observed(PRIOR, db)
    create = Source_code(PRIOR['VELOC_taup'])
    traces_obs, p_obs, s_obs, start_time_p, start_time_s = create.get_window_obspy(tr_obs, epi, depth, time,
                                                                                   VALUES['npts'])
    PRIOR['var_est'] = est_noise.get_var_data(start_time_p, tr_obs)
    # time_at_receiver = create.get_receiver_time(epi,depth, time)
    plt.figure()

    catalog_path = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/Additional_scripts/MQScatalog_withFrequencies/MQS_absolute_withFrequencyInfo.xml'
    catalog = Blindtest()
    events_catalog = catalog.get_events(catalog_path)

    for v in events_catalog:
        t, d, lat_ev, lo_ev = catalog.get_pref_origin(v)
        if time.date == t.date:
            Pick_event = v
            break
    PRIOR['M0'] = catalog.get_pref_scalarmoment(Pick_event)
    picks_surface = get_phase_picks(Pick_event, pick_type='surface')
    R_env_obs, L_env_obs = blindtest.pick_sw(tr_obs, picks_surface, epi, PRIOR, 30000, VALUES['directory'],
                                             plot_modus=False)

    start_sample = create_starting_sample()
    strike = np.random.uniform(PRIOR['strike']['range_min'], PRIOR['strike']['range_max'])
    dip = np.random.uniform(PRIOR['dip']['range_min'], PRIOR['dip']['range_max'])
    rake = np.random.uniform(PRIOR['rake']['range_min'], PRIOR['rake']['range_max'])
    sample_path = start_sample.get_sample_manual(epi, depth, strike, dip, rake, VALUES['directory'] + '/Blindtest_trialrun_sample.txt')
    # sample_path = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/Blindtest/Blindtest_trialrun_sample.txt'
    mcmc = MCMC_stream(R_env_obs=R_env_obs, L_env_obs=L_env_obs, total_traces_obs=traces_obs, P_traces_obs=p_obs,
                       S_traces_obs=s_obs, PRIOR=PRIOR, db=db, specification_values=VALUES, time_at_receiver=time,
                       start_sample_path=sample_path, picked_events=picks_surface, full_obs_trace=tr_obs,
                       P_start=start_time_p, S_start=start_time_s)

    mcmc.start_MCMC(VALUES['directory'] + '/Blindtest_trialrun.txt')

def Acces_Blindtest_check():
    BLINDTEST_MSEED = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/Database/data_Nienke/M5.0_3914855_deg_2019-09-22.mseed'
    BLINDTEST_XML = BLINDTEST_MSEED.replace(".mseed", ".xml")

    # Initiate Parameters:
    get_parameters = Get_Paramters()
    PRIOR = get_parameters.get_prior()
    VALUES = get_parameters.specifications()
    VALUES['npts'] = 2000
    VALUES['directory'] = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/Blindtest/check_waveforms'
    VALUES['blind'] = True

    # st = read(VALUES['directory'] + '/bw.mseed')
    # st_reject = read(VALUES['directory'] + '/bw_reject.mseed')

    # Initiate the databases from instaseis:
    db = instaseis.open_db(PRIOR['VELOC'])
    tr_obs = obspy.read(BLINDTEST_MSEED)
    # tr_obs.plot(outfile=VALUES['directory'] + '/Observed')
    tr_obs.integrate()
    tr_obs.plot(outfile=VALUES['directory'] + '/Observed_integrated')
    source = instaseis.Source.parse(BLINDTEST_XML)
    blindtest = Blindtest()
    events = blindtest.get_events(BLINDTEST_XML)
    # get_parameters.get_prior_blindtest(events[0])
    time, depth, la_s, lo_s = blindtest.get_pref_origin(events[0])


    dist, az, baz = gps2dist_azimuth(lat1=la_s,
                                     lon1=lo_s,
                                     lat2=PRIOR['la_r'],
                                     lon2=PRIOR['lo_r'], a=PRIOR['radius'], f=0)
    epi = kilometer2degrees(dist, radius=PRIOR['radius'])
    PRIOR['az'] = az
    PRIOR['baz'] = baz
    PRIOR['epi']['range_min'] = epi - 5
    PRIOR['epi']['range_max'] = epi + 5
    PRIOR['epi']['spread'] = 1
    PRIOR['depth']['range_min'] = depth - 10000
    PRIOR['depth']['range_max'] = depth + 10000
    PRIOR['network'] = tr_obs.traces[0].meta.network
    PRIOR['location'] = tr_obs.traces[0].meta.location
    PRIOR['station'] = tr_obs.traces[0].meta.station
    est_noise = Create_observed(PRIOR,db)
    create = Source_code(PRIOR['VELOC_taup'])
    traces_obs, p_obs, s_obs, p_time_obs, s_time_obs = create.get_window_obspy(tr_obs, epi, depth, time,
                                                                                   VALUES['npts'])
    PRIOR['var_est'] = est_noise.get_var_data(p_time_obs, tr_obs)
    obs_time = Create_observed(PRIOR,db)
    time_at_receiver = obs_time.get_receiver_time(epi, depth, time)
    plt.figure()

    catalog_path = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/Additional_scripts/MQScatalog_withFrequencies/MQS_absolute_withFrequencyInfo.xml'

    events_catalog = blindtest.get_events(catalog_path)

    for v in events_catalog:
        t, d, lat_ev, lo_ev = blindtest.get_pref_origin(v)
        if time.date == t.date:
            Pick_event = v
            break
    PRIOR['M0'] = blindtest.get_pref_scalarmoment(Pick_event)
    picks_surface = get_phase_picks(Pick_event, pick_type='surface')
    R_env_obs, L_env_obs = blindtest.pick_sw(tr_obs, picks_surface, epi, PRIOR, VALUES['npts'], VALUES['directory'],
                                             plot_modus=True)

    start_sample = create_starting_sample()
    strike = 243.423396191
    dip =34.436087773
    rake = 164.912874159


    from Seismogram import Seismogram
    from Misfit import Misfit
    misfit = Misfit(VALUES['directory'])
    seis = Seismogram(PRIOR,db)
    epi = epi -3
    depth = depth

    # ---------------------------------------------------------------------------------------------------------------  #
    dict = geo.Geodesic(a=PRIOR['radius'], f=0).ArcDirect(lat1=PRIOR['la_r'], lon1=PRIOR['lo_r'],
                                                               azi1=PRIOR['baz'],
                                                               a12=epi, outmask=1929)
    d_syn, traces_syn, sources = seis.get_seis_manual(la_s=dict['lat2'], lo_s=dict['lon2'], depth=depth,
                                                           strike=strike, dip=dip, rake=rake,
                                                           time=time, M0=PRIOR['M0'],sdr=VALUES['sdr'])

    R_env_syn, L_env_syn = blindtest.pick_sw(traces_syn, picks_surface, epi, PRIOR, VALUES['npts'], VALUES['directory'],
                                               plot_modus=False)

    traces_syn.plot(outfile=VALUES['directory'] + '/syntethic')
    total_syn, p_syn, s_syn, p_time_syn, s_time_syn = create.get_window_obspy(traces_syn, epi, depth,time, VALUES['npts'])

    ax1 = plt.subplot2grid((5, 1), (0, 0))
    ax1.plot(zero_to_nan(p_syn.traces[0].data), c='r', linewidth=0.3)
    ax1.plot(zero_to_nan(p_obs.traces[0].data),c='k',linestyle=':', linewidth=0.3)
    plt.tight_layout()
    ax2 = plt.subplot2grid((5, 1), (1, 0))
    ax2.plot(zero_to_nan(p_syn.traces[1].data), c='r', linewidth=0.3)
    ax2.plot(zero_to_nan(p_obs.traces[1].data),c='k',linestyle=':', linewidth=0.3)
    plt.tight_layout()
    ax3 = plt.subplot2grid((5, 1), (2, 0))
    ax3.plot(zero_to_nan(s_syn.traces[0].data), c='r', linewidth=0.3)
    ax3.plot(zero_to_nan(s_obs.traces[0].data),c='k', linewidth=0.3)
    plt.tight_layout()
    ax4 = plt.subplot2grid((5, 1), (3, 0))
    ax4.plot(zero_to_nan(s_syn.traces[1].data), c='r', linewidth=0.3)
    ax4.plot(zero_to_nan(s_obs.traces[1].data),c='k', linewidth=0.3)
    plt.tight_layout()
    ax5 = plt.subplot2grid((5, 1), (4, 0))
    ax5.plot(zero_to_nan(s_syn.traces[2].data), c='r', linewidth=0.3)
    ax5.plot(zero_to_nan(s_obs.traces[2].data),c='k', linewidth=0.3)
    plt.tight_layout()

    plt.savefig(VALUES['directory'] + '/%.2f_%.2f.pdf' %(epi,depth))
    plt.close()

    # time =

    ax1 = plt.subplot2grid((3, 1), (0, 0))
    ax1.plot(zero_to_nan(total_syn.traces[0].data), c='r', linewidth=0.5)
    ax1.plot(zero_to_nan(traces_obs.traces[0].data),c='k',linestyle=':', linewidth=0.5)
    ax1.set_title('SYNTHETIC: = epi: %.2f  REAL: epi = %.2f (depth fixed' %(epi,epi+3))
    plt.tight_layout()
    ax2 = plt.subplot2grid((3, 1), (1, 0))
    ax2.plot(zero_to_nan(total_syn.traces[1].data), c='r', linewidth=0.5)
    ax2.plot(zero_to_nan(traces_obs.traces[1].data),c='k',linestyle=':', linewidth=0.5)
    plt.tight_layout()
    ax3 = plt.subplot2grid((3, 1), (2, 0))
    ax3.plot(zero_to_nan(total_syn.traces[2].data), c='r', linewidth=0.5)
    ax3.plot(zero_to_nan(traces_obs.traces[2].data),c='k',linestyle=':', linewidth=0.5)
    plt.tight_layout()


    plt.savefig(VALUES['directory'] + '/PS_%.2f_%.2f.pdf' %(epi,depth))
    plt.close()

    Xi_bw_new, time_shift_new, amplitude = misfit.CC_stream(p_obs, p_syn, s_obs,s_syn,p_time_obs,p_time_syn)
    s_z_new = 0.1 * Xi_bw_new[0]
    s_r_new = 0.1 * Xi_bw_new[1]
    s_t_new = 1 * Xi_bw_new[2]
    p_z_new = 5 * Xi_bw_new[3]
    p_r_new = 5 * Xi_bw_new[4]
    bw_new = s_z_new + s_r_new + s_t_new + p_z_new + p_r_new
    Xi_R_new = misfit.SW_L2(R_env_obs, R_env_syn, PRIOR['var_est'],amplitude)
    Xi_L_new = misfit.SW_L2(L_env_obs, L_env_syn, PRIOR['var_est'],amplitude)

    R_dict_new = {}
    rw_new = 0
    for j, v in enumerate(Xi_R_new):
        R_dict_new.update({'R_%i_new' % j:  v})
        rw_new +=  v

    L_dict_new = {}
    lw_new = 0
    for j, v in enumerate(Xi_L_new):
        L_dict_new.update({'L_%i_new' % j:  v})
        lw_new += v
    Xi_new = bw_new + rw_new + lw_new
    a=1

def Normal_check():
    # Initiate Parameters:

    get_parameters = Get_Paramters()
    PARAMETERS = get_parameters.get_unkown()
    PRIOR = get_parameters.get_prior()
    VALUES = get_parameters.specifications()
    VALUES['blind'] = False
    VALUES['directory'] = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/Final/check_waveforms'

    ## DISCUSS THIS!!!!
    PRIOR['az'] = PARAMETERS['az']
    PRIOR['baz'] = PARAMETERS['baz']

    # Initiate the databases from instaseis:
    db = instaseis.open_db(PRIOR['VELOC'])
    create = Create_observed(PRIOR, db)

    d_obs, tr_obs, source = create.get_seis_automatic(parameters=PARAMETERS, prior=PRIOR, noise_model=VALUES['noise'],
                                                      sdr=VALUES['sdr'])
    traces_obs, p_obs, s_obs ,p_time_obs, s_time_obs= create.get_window_obspy(tr_obs, PARAMETERS['epi'], PARAMETERS['depth_s'],
                                                       PARAMETERS['origin_time'], VALUES['npts'])
    time_at_receiver = create.get_receiver_time(PARAMETERS['epi'], PARAMETERS['depth_s'], PARAMETERS['origin_time'])

    PRIOR['var_est'] = create.get_var_data(p_time_obs,tr_obs)


    sw = Surface_waves(PRIOR)
    R_env_obs = sw.rayleigh_pick(tr_obs.traces[0], PARAMETERS['la_s'], PARAMETERS['lo_s'], PARAMETERS['depth_s'],
                                 VALUES['directory'], PARAMETERS['origin_time'], VALUES['npts'], plot_modus=False)
    L_env_obs = sw.love_pick(tr_obs.traces[2], PARAMETERS['la_s'], PARAMETERS['lo_s'], PARAMETERS['depth_s'],
                             VALUES['directory'], PARAMETERS['origin_time'], VALUES['npts'], plot_modus=False)
    # tr_obs.plot()
    # ------------------------------------------------------------------

    seis = Seismogram(PRIOR, db)
    window_code = Source_code(PRIOR['VELOC_taup'])
    misfit = Misfit(VALUES['directory'])

    epi = PARAMETERS['epi']
    depth = PARAMETERS['depth_s']
    strike = PARAMETERS['strike']
    dip = PARAMETERS['dip']
    rake = PARAMETERS['rake']
    time = PARAMETERS['origin_time']

    # ---------------------------------------------------------------------------------------------------------------  #
    dict = geo.Geodesic(a=PRIOR['radius'], f=0).ArcDirect(lat1=PRIOR['la_r'], lon1=PRIOR['lo_r'],
                                                               azi1=PRIOR['baz'],
                                                               a12=epi, outmask=1929)
    d_syn, traces_syn, sources = seis.get_seis_manual(la_s=dict['lat2'], lo_s=dict['lon2'], depth=depth,
                                                           strike=strike, dip=dip, rake=rake,
                                                           time=time, M0=PRIOR['M0'],sdr=VALUES['sdr'])

    R_env_syn = sw.rayleigh_pick(Z_trace=traces_syn.traces[0], la_s=dict['lat2'], lo_s=dict['lon2'],
                                      depth=depth, save_directory=VALUES['directory'], time_at_rec=time,
                                      npts=VALUES['npts'], plot_modus=True)
    L_env_syn = sw.love_pick(T_trace=traces_syn.traces[2], la_s=dict['lat2'], lo_s=dict['lon2'],
                                  depth=depth, save_directory=VALUES['directory'], time_at_rec=time,
                                  npts=VALUES['npts'], plot_modus=False)

    traces_syn.plot(outfile=VALUES['directory'] + '/syntethic')
    total_syn, p_syn, s_syn, p_time_syn, s_time_syn = window_code.get_window_obspy(traces_syn, epi, depth,time, VALUES['npts'])

    ax1 = plt.subplot2grid((5, 1), (0, 0))
    ax1.plot(zero_to_nan(p_syn.traces[0].data), c='r', linewidth=0.3)
    ax1.plot(zero_to_nan(p_obs.traces[0].data),c='k',linestyle=':', linewidth=0.3)
    plt.tight_layout()
    ax2 = plt.subplot2grid((5, 1), (1, 0))
    ax2.plot(zero_to_nan(p_syn.traces[1].data), c='r', linewidth=0.3)
    ax2.plot(zero_to_nan(p_obs.traces[1].data),c='k',linestyle=':', linewidth=0.3)
    plt.tight_layout()
    ax3 = plt.subplot2grid((5, 1), (2, 0))
    ax3.plot(zero_to_nan(s_syn.traces[0].data), c='r', linewidth=0.3)
    ax3.plot(zero_to_nan(s_obs.traces[0].data),c='k', linewidth=0.3)
    plt.tight_layout()
    ax4 = plt.subplot2grid((5, 1), (3, 0))
    ax4.plot(zero_to_nan(s_syn.traces[1].data), c='r', linewidth=0.3)
    ax4.plot(zero_to_nan(s_obs.traces[1].data),c='k', linewidth=0.3)
    plt.tight_layout()
    ax5 = plt.subplot2grid((5, 1), (4, 0))
    ax5.plot(zero_to_nan(s_syn.traces[2].data), c='r', linewidth=0.3)
    ax5.plot(zero_to_nan(s_obs.traces[2].data),c='k', linewidth=0.3)
    plt.tight_layout()

    plt.savefig(VALUES['directory'] + '/%.2f_%.2f.pdf' %(epi,depth))
    plt.close()

    ax1 = plt.subplot2grid((3, 1), (0, 0))
    ax1.plot(zero_to_nan(total_syn.traces[0].data), c='r', linewidth=0.5)
    ax1.plot(zero_to_nan(traces_obs.traces[0].data),c='k',linestyle=':', linewidth=0.5)
    ax1.set_title('SYNTHETIC: = epi: %.2f  REAL: epi = %.2f (depth fixed' %(epi,epi))
    plt.tight_layout()
    ax2 = plt.subplot2grid((3, 1), (1, 0))
    ax2.plot(zero_to_nan(total_syn.traces[1].data), c='r', linewidth=0.5)
    ax2.plot(zero_to_nan(traces_obs.traces[1].data),c='k',linestyle=':', linewidth=0.5)
    plt.tight_layout()
    ax3 = plt.subplot2grid((3, 1), (2, 0))
    ax3.plot(zero_to_nan(total_syn.traces[2].data), c='r', linewidth=0.5)
    ax3.plot(zero_to_nan(traces_obs.traces[2].data),c='k',linestyle=':', linewidth=0.5)
    plt.tight_layout()


    plt.savefig(VALUES['directory'] + '/PS_%.2f_%.2f.pdf' %(epi,depth))
    plt.close()

    Xi_bw_new, time_shift_new, amplitude = misfit.CC_stream(p_obs, p_syn, s_obs,s_syn,p_time_obs,p_time_syn)
    s_z_new = 0.1 * Xi_bw_new[0]
    s_r_new = 0.1 * Xi_bw_new[1]
    s_t_new = 1 * Xi_bw_new[2]
    p_z_new = 5 * Xi_bw_new[3]
    p_r_new = 5 * Xi_bw_new[4]
    bw_new = s_z_new + s_r_new + s_t_new + p_z_new + p_r_new
    Xi_R_new = misfit.SW_L2(R_env_obs, R_env_syn, PRIOR['var_est'],amplitude)
    Xi_L_new = misfit.SW_L2(L_env_obs, L_env_syn, PRIOR['var_est'],amplitude)

    R_dict_new = {}
    rw_new = 0
    for j, v in enumerate(Xi_R_new):
        R_dict_new.update({'R_%i_new' % j:  v})
        rw_new +=  v

    L_dict_new = {}
    lw_new = 0
    for j, v in enumerate(Xi_L_new):
        L_dict_new.update({'L_%i_new' % j:  v})
        lw_new += v
    Xi_new = bw_new + rw_new + lw_new
    a=1

def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x == 0 else x for x in values]

if __name__ == '__main__':
    main()


