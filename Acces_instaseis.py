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

    ## DISCUSS THIS!!!!
    PRIOR['az'] = PARAMETERS['az']
    PRIOR['baz'] = PARAMETERS['baz']

    # Initiate the databases from instaseis:
    db = instaseis.open_db(PRIOR['VELOC'])
    create = Create_observed(PRIOR, db)

    d_obs, tr_obs, source = create.get_seis_automatic(parameters=PARAMETERS, prior=PRIOR, noise_model=VALUES['noise'],
                                                      sdr=VALUES['sdr'])
    traces_obs, p_obs, s_obs = create.get_window_obspy(tr_obs, PARAMETERS['epi'], PARAMETERS['depth_s'],
                                                       PARAMETERS['origin_time'], VALUES['npts'])
    time_at_receiver = create.get_receiver_time(PARAMETERS['epi'], PARAMETERS['depth_s'], PARAMETERS['origin_time'])

    # create.get_fft(traces=traces, directory=VALUES['directory'])

    sw = Surface_waves(PRIOR)
    R_env_obs = sw.rayleigh_pick(tr_obs.traces[0], PARAMETERS['la_s'], PARAMETERS['lo_s'], PARAMETERS['depth_s'],
                                 VALUES['directory'], PARAMETERS['origin_time'], VALUES['npts'], plot_modus=True)
    L_env_obs = sw.love_pick(tr_obs.traces[2], PARAMETERS['la_s'], PARAMETERS['lo_s'], PARAMETERS['depth_s'],
                             VALUES['directory'], PARAMETERS['origin_time'], VALUES['npts'], plot_modus=True)
    # tr_obs.plot()
    # ------------------------------------------------------------------

    seis = Seismogram(PRIOR, db)
    window_code = Source_code(PRIOR['VELOC_taup'])
    misfit = Misfit(VALUES['directory'])

    start_sample_path = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/Final/close_sample.txt'
    # start_sample_path = None

    m = MCMC_stream(R_env_obs, L_env_obs, traces_obs, p_obs, s_obs, PRIOR, db, VALUES, time_at_receiver,
                    start_sample_path)
    m.start_MCMC(VALUES['directory'] + '/New_misfit.txt')


def Acces_Blindtest():
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
    tr_obs.plot(outfile=VALUES['directory'] + '/Observed')
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

    create = Source_code(PRIOR['VELOC_taup'])
    traces_obs, p_obs, s_obs, start_time_p, start_time_s = create.get_window_obspy(tr_obs, epi, depth, time,
                                                                                   VALUES['npts'])
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
                                             plot_modus=True)

    start_sample = create_starting_sample()
    strike = np.random.uniform(PRIOR['strike']['range_min'], PRIOR['strike']['range_max'])
    dip = np.random.uniform(PRIOR['dip']['range_min'], PRIOR['dip']['range_max'])
    rake = np.random.uniform(PRIOR['rake']['range_min'], PRIOR['rake']['range_max'])
    sample_path = start_sample.get_sample_manual(epi, depth, strike, dip, rake, VALUES['directory'] + '/Pref_start.txt')
    mcmc = MCMC_stream(R_env_obs=R_env_obs, L_env_obs=L_env_obs, total_traces_obs=traces_obs, P_traces_obs=p_obs,
                       S_traces_obs=s_obs, PRIOR=PRIOR, db=db, specification_values=VALUES, time_at_receiver=time,
                       start_sample_path=sample_path, picked_events=picks_surface, full_obs_trace=tr_obs,
                       P_start=start_time_p, S_start=start_time_s)

    mcmc.start_MCMC(VALUES['directory'] + '/Blindtest_trialrun.txt')


if __name__ == '__main__':
    main()


