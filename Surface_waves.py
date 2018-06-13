import instaseis
import numpy as np
from obspy.signal.filter import envelope
from obspy.geodetics import kilometer2degrees
from obspy.geodetics.base import gps2dist_azimuth
from obspy.core.stream import Stream
from obspy.core.trace import Trace
import matplotlib.pylab as plt
from scipy.signal import hilbert
import os

## All different classes:
from Get_Parameters import Get_Paramters
from Create_observed import Create_observed

def main():

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
    create.get_fft(traces=traces,directory=VALUES['directory'])

    sw = Surface_waves(PRIOR)
    env_trace = sw.rayleigh_pick(traces.traces[0],PARAMETERS['la_s'], PARAMETERS['lo_s'], PARAMETERS['depth_s'], VALUES['directory'],PARAMETERS['origin_time'])

class Surface_waves:

    def __init__(self, PRIOR):
        self.prior = PRIOR

    def get_khan_phases(self,time_at_rec):
        phases = []
        phases.append(dict(starttime=lambda dist, depth: time_at_rec + dist / 2.8,
                           endtime=lambda dist, depth: time_at_rec + dist / 2.6,
                           comp='Z',
                           fmin=1. / 20.,
                           fmax=1. / 10.,
                           dt=5.0,
                           name='R1_10_20'))
        phases.append(dict(starttime=lambda dist, depth:time_at_rec+ dist / 2.8,
                           endtime=lambda dist, depth: time_at_rec + dist / 2.6,
                           comp='Z',
                           fmin=1. / 16.,
                           fmax=1. / 8.,
                           dt=4.0,
                           name='R1_08_16'))
        phases.append(dict(starttime=lambda dist, depth: time_at_rec + dist / 2.8,  # 2.7,
                           endtime=lambda dist, depth:time_at_rec + dist / 2.5,  # 2.5,
                           comp='Z',
                           fmin=1. / 32.,
                           fmax=1. / 16.,
                           dt=8.0,
                           name='R1_16_32'))
        phases.append(dict(starttime=lambda dist, depth: time_at_rec + dist / 2.8,  # /2.7,
                           endtime=lambda dist, depth: time_at_rec + dist / 2.4,  # /2.5,
                           comp='Z',
                           fmin=1. / 48.,
                           fmax=1. / 24.,
                           dt=12.0,
                           name='R1_24_48'))
        return phases

    def get_R_phases(self,time_at_rec):
        phases = []
        phases.append(dict(starttime=lambda dist, depth: time_at_rec + dist / 2.8,
                           endtime=lambda dist, depth: time_at_rec + dist / 2.6,
                           comp='Z',
                           fmin=1. / 20.,
                           fmax=1. / 10.,
                           dt=5.0,
                           name='R1_10_20'))
        phases.append(dict(starttime=lambda dist, depth:time_at_rec+ dist / 2.8,
                           endtime=lambda dist, depth: time_at_rec + dist / 2.6,
                           comp='Z',
                           fmin=1. / 16.,
                           fmax=1. / 8.,
                           dt=4.0,
                           name='R1_08_16'))
        phases.append(dict(starttime=lambda dist, depth: time_at_rec + dist / 2.8,  # 2.7,
                           endtime=lambda dist, depth:time_at_rec + dist / 2.5,  # 2.5,
                           comp='Z',
                           fmin=1. / 32.,
                           fmax=1. / 16.,
                           dt=8.0,
                           name='R1_16_32'))
        phases.append(dict(starttime=lambda dist, depth: time_at_rec + dist / 2.8,  # /2.7,
                           endtime=lambda dist, depth: time_at_rec + dist / 2.4,  # /2.5,
                           comp='Z',
                           fmin=1. / 48.,
                           fmax=1. / 24.,
                           dt=12.0,
                           name='R1_24_48'))
        return phases

    def get_L_phases(self,time_at_rec):
        phases = []
        phases.append(dict(starttime=lambda dist, depth: time_at_rec+ dist / 3.2,  # /3.15,
                           endtime=lambda dist, depth:time_at_rec + dist / 2.85,  # /2.95,
                           comp='T',
                           fmin=1. / 48.,
                           fmax=1. / 24.,
                           dt=5.0,
                           name='G1_24_48'))
        phases.append(dict(starttime=lambda dist, depth:time_at_rec + dist / 3.25,  # /3.2,
                           endtime=lambda dist, depth: time_at_rec + dist / 2.9,  # /2.95,
                           comp='T',
                           fmin=1. / 32.,
                           fmax=1. / 16.,
                           dt=8.0,
                           name='G1_16_32'))
        phases.append(dict(starttime=lambda dist, depth:time_at_rec + dist / 3.2,  # /3.15,
                           endtime=lambda dist, depth: time_at_rec + dist / 2.9,  # /2.95,
                           comp='T',
                           fmin=1. / 24.,
                           fmax=1. / 12.,
                           dt=6.0,
                           name='G1_12_24'))
        phases.append(dict(starttime=lambda dist, depth: time_at_rec+ dist / 3.15,  # /3.1,
                           endtime=lambda dist, depth: time_at_rec + dist / 2.9,  # /3.00,
                           comp='T',
                           fmin=1. / 16.,
                           fmax=1. / 8.,
                           dt=4.0,
                           name='G1_08_16'))
        return phases

    def rayleigh_pick(self, Z_trace, la_s, lo_s,depth ,save_directory, time_at_rec):
        dir_R = save_directory + '/Rayleigh_waves'
        if not os.path.exists(dir_R):
            os.makedirs(dir_R)
        Rayleigh_st = Stream()

        evla = la_s
        evlo = lo_s

        rec = instaseis.Receiver(latitude=self.prior['la_r'], longitude=self.prior['lo_r'])

        dist, az, baz = gps2dist_azimuth(lat1=evla,
                                         lon1=evlo,
                                         lat2=self.prior['la_r'],
                                         lon2=self.prior['lo_r'], a=self.prior['radius'], f=0)

        # For now I am just using the Z-component, because this will have the strongest Rayleigh signal:
        Z_comp = Z_trace.copy()

        Z_comp.plot(outfile=dir_R+'/sw_entire_waveform.pdf')
        phases = self.get_R_phases(time_at_rec)

        for i in range(len(phases)):
            dir_phases = dir_R + '/%s'%phases[i]['name']
            if not os.path.exists(dir_phases):
                os.makedirs(dir_phases)
            trial = Z_trace.copy()
            trial.detrend(type="demean")
            trial.interpolate(sampling_rate = 10./phases[i]['dt']) # No method specified, so : 'weighted_average_slopes' is used
            # trial.filter('highpass', freq=phases[i]['fmin'], zerophase=True)
            # trial.filter('lowpass', freq=phases[i]['fmax'], zerophase=True)
            trial.filter('bandpass',freqmin=phases[i]['fmin'],freqmax=phases[i]['fmax'],zerophase = True)
            trial.detrend()

            ax = plt.subplot(111)
            plt.plot(trial.data,alpha = 0.5)
            start_vline = int((phases[i]['starttime'](dist, depth).timestamp - time_at_rec.timestamp) / trial.stats.delta)
            end_vline = int((phases[i]['endtime'](dist, depth).timestamp - time_at_rec.timestamp) / trial.stats.delta)
            ymin,ymax = ax.get_ylim()
            # plt.plot(trial.data)
            plt.vlines([start_vline, end_vline],ymin,ymax)
            plt.xlabel(time_at_rec.strftime('%Y-%m-%dT%H:%M:%S + sec'))
            plt.savefig(dir_phases + '/sw_with_Rayleigh_windows.pdf')
            plt.tight_layout()
            plt.close()


            trial.trim(starttime=phases[i]['starttime'](dist, depth), endtime=phases[i]['endtime'](dist, depth))

            trial.detrend(type="demean")

            trial.taper(max_percentage=0.2) # No type specified, so at this moment using a Hann type
            trial.decimate(10)

            env = envelope(trial.data)
            analytical_signal = hilbert(trial.data)
            amplitude_envelope = np.abs(analytical_signal)
            plt.plot(trial)
            plt.plot(env)
            plt.plot(amplitude_envelope,':')
            plt.savefig(dir_phases + '/Rayleigh_envelope_filter_%s.pdf' %phases[i]['name'])
            plt.close()
            env_trace = Trace(env,
                               header={"starttime": phases[i]['starttime'](dist, depth) , 'delta': trial.meta.delta, "station": trial.meta.station,
                                       "network": trial.meta.network, "location": trial.meta.location,
                                       "channel": trial.meta.channel, "instaseis": trial.meta.instaseis})
            Rayleigh_st.append(env_trace)
        return Rayleigh_st



if __name__ == '__main__':
    main()
