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

class Surface_waves:

    def __init__(self, PRIOR):
        self.prior = PRIOR

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
                           endtime=lambda dist, depth:time_at_rec + dist / 2.6,  # 2.5,
                           comp='Z',
                           fmin=1. / 32.,
                           fmax=1. / 16.,
                           dt=8.0,
                           name='R1_16_32'))
        phases.append(dict(starttime=lambda dist, depth: time_at_rec + dist / 2.8,  # /2.7,
                           endtime=lambda dist, depth: time_at_rec + dist / 2.5,  # /2.5,
                           comp='Z',
                           fmin=1. / 48.,
                           fmax=1. / 24.,
                           dt=12.0,
                           name='R1_24_48'))
        return phases

    def get_L_phases(self,time_at_rec):
        phases = []
        phases.append(dict(starttime=lambda dist, depth: time_at_rec+ dist / 3.2,  # /3.15,
                           endtime=lambda dist, depth:time_at_rec + dist / 2.9,  # /2.95,
                           comp='T',
                           fmin=1. / 48.,
                           fmax=1. / 24.,
                           dt=5.0,
                           name='G1_24_48'))
        phases.append(dict(starttime=lambda dist, depth:time_at_rec + dist / 3.2,  # /3.2,
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
        phases.append(dict(starttime=lambda dist, depth: time_at_rec+ dist / 3.2,  # /3.1,
                           endtime=lambda dist, depth: time_at_rec + dist / 2.9,  # /3.00,
                           comp='T',
                           fmin=1. / 16.,
                           fmax=1. / 8.,
                           dt=4.0,
                           name='G1_08_16'))
        return phases


    def rayleigh_pick(self, Z_trace, la_s, lo_s, depth, save_directory, time_at_rec, npts,filter = True ,plot_modus = False):
        if plot_modus == True:
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

        if plot_modus == True:
            Z_comp.plot(outfile=dir_R + '/sw_entire_waveform.pdf')
        phases = self.get_R_phases(time_at_rec)

        for i in range(len(phases)):
            if plot_modus == True:
                dir_phases = dir_R + '/%s' % phases[i]['name']
                if not os.path.exists(dir_phases):
                    os.makedirs(dir_phases)
            trial = Z_trace.copy()
            if filter == True:
                trial.detrend(type="demean")
                trial.filter('highpass', freq=phases[i]['fmin'], zerophase=True)
                trial.filter('lowpass', freq=phases[i]['fmax'], zerophase=True)
                trial.detrend()

            if plot_modus == True:
                start_vline = int(
                    (phases[i]['starttime'](dist, depth).timestamp - time_at_rec.timestamp) / trial.stats.delta)
                end_vline = int(
                    (phases[i]['endtime'](dist, depth).timestamp - time_at_rec.timestamp) /trial.stats.delta)
                plt.figure(4)
                ax = plt.subplot(111)
                plt.plot(trial.data, alpha=0.5)
                ymin, ymax = ax.get_ylim()
                plt.plot(trial.data)
                plt.vlines([start_vline, end_vline], ymin, ymax)
                plt.xlabel(time_at_rec.strftime('%Y-%m-%dT%H:%M:%S + sec'))
                plt.tight_layout()
                plt.savefig(dir_phases + '/sw_with_Rayleigh_windows.pdf')
                # plt.show()
                plt.close()

            if filter == True:
                trial.detrend(type="demean")
                env = envelope(trial.data)
                trial.data = env
                trial.trim(starttime=phases[i]['starttime'](dist, depth), endtime=phases[i]['endtime'](dist, depth))

            else:
                env = trial.data
            if plot_modus == True:
                plt.figure(5)
                plt.plot(trial,label = '%s' % phases[i]['name'])
                plt.legend()
                plt.tight_layout()
                plt.savefig(dir_phases + '/Rayleigh_envelope_filter_%s.pdf' % phases[i]['name'])
                # plt.show()
                plt.close()

            zero_trace = Trace(np.zeros(npts),
                               header={"starttime": phases[i]['starttime'](dist, depth), 'delta': trial.meta.delta,
                                       "station": trial.meta.station,
                                       "network": trial.meta.network, "location": trial.meta.location,
                                       "channel": phases[i]['name']})
            total_trace = zero_trace.__add__(trial, method=0, interpolation_samples=0,
                                             fill_value=trial.data,
                                             sanity_checks=False)
            Rayleigh_st.append(total_trace)
        if plot_modus == True:
            plt.figure(6)
            plt.plot(Rayleigh_st.traces[0].data, label='%s' % Rayleigh_st.traces[0].meta.channel)
            plt.plot(Rayleigh_st.traces[1].data, label='%s' % Rayleigh_st.traces[1].meta.channel)
            plt.plot(Rayleigh_st.traces[2].data, label='%s' % Rayleigh_st.traces[2].meta.channel)
            plt.plot(Rayleigh_st.traces[3].data, label='%s' % Rayleigh_st.traces[3].meta.channel)
            plt.legend()
            plt.tight_layout()
            plt.savefig(dir_R + '/diff_Rayleigh_freq.pdf')
            plt.close()
        return Rayleigh_st

    def love_pick(self, T_trace, la_s, lo_s, depth, save_directory, time_at_rec, npts, filter = True, plot_modus= False):
        if plot_modus == True:
            dir_L = save_directory + '/Love_waves'
            if not os.path.exists(dir_L):
                os.makedirs(dir_L)
        Love_st = Stream()

        evla = la_s
        evlo = lo_s

        rec = instaseis.Receiver(latitude=self.prior['la_r'], longitude=self.prior['lo_r'])

        dist, az, baz = gps2dist_azimuth(lat1=evla,
                                         lon1=evlo,
                                         lat2=self.prior['la_r'],
                                         lon2=self.prior['lo_r'], a=self.prior['radius'], f=0)

        # For now I am just using the Z-component, because this will have the strongest Rayleigh signal:
        T_comp = T_trace.copy()

        if plot_modus == True:
            T_comp.plot(outfile=dir_L + '/sw_entire_waveform.pdf')
        phases = self.get_L_phases(time_at_rec)

        for i in range(len(phases)):
            if plot_modus == True:
                dir_phases = dir_L + '/%s' % phases[i]['name']
                if not os.path.exists(dir_phases):
                    os.makedirs(dir_phases)
            trial = T_trace.copy()
            if filter == True:
                trial.detrend(type="demean")
                trial.filter('highpass', freq=phases[i]['fmin'], zerophase=True)
                trial.filter('lowpass', freq=phases[i]['fmax'], zerophase=True)
                trial.detrend()

            if plot_modus == True:
                start_vline = int(
                    (phases[i]['starttime'](dist, depth).timestamp - time_at_rec.timestamp) / trial.stats.delta)
                end_vline = int(
                    (phases[i]['endtime'](dist, depth).timestamp - time_at_rec.timestamp) / trial.stats.delta)
                plt.figure(1)
                ax=plt.subplot(111)
                plt.plot(trial.data, alpha=0.5)
                ymin, ymax = ax.get_ylim()
                # plt.plot(trial.data)
                plt.vlines([start_vline, end_vline], ymin, ymax)
                plt.xlabel(time_at_rec.strftime('%Y-%m-%dT%H:%M:%S + sec'))
                plt.savefig(dir_phases + '/sw_with_Love_windows.pdf')
                plt.tight_layout()
                plt.close()

            if filter == True:
                trial.detrend(type="demean")
                env = envelope(trial.data)
                trial.data = env
                trial.trim(starttime=phases[i]['starttime'](dist, depth), endtime=phases[i]['endtime'](dist, depth))
            else:
                env = trial.data
            if plot_modus == True:
                plt.figure(2)
                plt.plot(trial,label = '%s' % phases[i]['name'])
                plt.legend()
                plt.tight_layout()
                plt.savefig(dir_phases + '/Love_envelope_filter_%s.pdf' % phases[i]['name'])
                plt.close()

            zero_trace = Trace(np.zeros(npts),
                               header={"starttime": phases[i]['starttime'](dist, depth), 'delta': trial.meta.delta,
                                       "station": trial.meta.station,
                                       "network": trial.meta.network, "location": trial.meta.location,
                                       "channel": phases[i]['name']})

            total_trace = zero_trace.__add__(trial, method=0, interpolation_samples=0,
                                             fill_value=trial.data,
                                             sanity_checks=False)

            Love_st.append(total_trace)
        if plot_modus == True:
            plt.figure(3)
            plt.plot(Love_st.traces[0].data, label='%s' % Love_st.traces[0].meta.channel)
            plt.plot(Love_st.traces[1].data, label='%s' % Love_st.traces[1].meta.channel)
            plt.plot(Love_st.traces[2].data, label='%s' % Love_st.traces[2].meta.channel)
            plt.plot(Love_st.traces[3].data, label='%s' % Love_st.traces[3].meta.channel)
            plt.legend()
            plt.tight_layout()
            plt.savefig(dir_L + '/diff_Love_freq.pdf')
            plt.close()
        return Love_st

    def filter(self,stream,time_at_rec,la_s,lo_s,depth,Rayleigh = True):
        env_stream = Stream()
        dist, az, baz = gps2dist_azimuth(lat1=la_s,
                                         lon1=lo_s,
                                         lat2=self.prior['la_r'],
                                         lon2=self.prior['lo_r'], a=self.prior['radius'], f=0)
        if Rayleigh == True:
            phases = self.get_R_phases(time_at_rec)
        else:
            phases = self.get_L_phases(time_at_rec)
        for i,v in enumerate(stream.traces):
            npts = len(v.data)
            trace = stream.traces[i].copy()
            trace.detrend(type="demean")
            trace.interpolate(
                sampling_rate=10. / phases[i]['dt'])  # No method specified, so : 'weighted_average_slopes' is used
            trace.filter('highpass', freq=phases[i]['fmin'], zerophase=True)
            trace.filter('lowpass', freq=phases[i]['fmax'], zerophase=True)
            trace.detrend()
            trace.detrend(type="demean")
            env = envelope(trace.data)

            zero_trace = Trace(np.zeros(npts),
                               header={"starttime": phases[i]['starttime'](dist, depth), 'delta': trace.meta.delta,
                                       "station": trace.meta.station,
                                       "network": trace.meta.network, "location": trace.meta.location,
                                       "channel": trace.meta.channel, "instaseis": trace.meta.instaseis})
            env_trace = Trace(env,
                              header={"starttime": phases[i]['starttime'](dist, depth), 'delta': trace.meta.delta,
                                      "station": trace.meta.station,
                                      "network": trace.meta.network, "location": trace.meta.location,
                                      "channel": trace.meta.channel, "instaseis": trace.meta.instaseis})

            env_stream.append(env_trace)

        return env_stream






