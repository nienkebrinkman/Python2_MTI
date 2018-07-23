import instaseis
import seismogram_noise
from obspy.taup import TauPyModel
import obspy
import numpy as np
from obspy.core.stream import Stream
from obspy.core.trace import Trace
import matplotlib.pylab as plt
from obspy.signal.filter import envelope

class Create_observed:
    def __init__(self, PRIOR, db):
        self.prior = PRIOR
        self.db = db
        self.veloc_model = self.prior['VELOC_taup']
    def get_receiver(self):
        receiver = instaseis.Receiver(latitude=self.prior['la_r'], longitude=self.prior['lo_r'],
                                      network=self.prior['network'], station=self.prior['station'])
        return receiver

    def get_source(self, la_s, lo_s, depth, strike=None, dip=None, rake=None, m_tp=None, m_rp=None, m_rt=None,
                   m_pp=None, m_tt=None, m_rr=None, time=None, sdr=False):
        if sdr == True:
            source = instaseis.Source.from_strike_dip_rake(latitude=la_s, longitude=lo_s,
                                                           depth_in_m=depth,
                                                           strike=strike, dip=dip,
                                                           rake=rake, M0=self.prior['M0'],
                                                           origin_time=time,dt = 2.0)
            return source
        else:
            source = instaseis.Source(latitude=la_s, longitude=lo_s,
                                           depth_in_m=depth, m_tp=m_tp, m_rp=m_rp,
                                           m_rt=m_rt, m_pp=m_pp, m_tt=m_tt,
                                           m_rr=m_rr, origin_time=time)
            return source

    def get_seis_automatic(self, parameters,prior, noise_model=False, sdr=False):
        self.prior = prior
        if sdr == True:
            source = self.get_source(la_s=parameters['la_s'], lo_s=parameters['lo_s'], depth=parameters['depth_s'],
                                     strike=parameters['strike'], dip=parameters['dip'], rake=parameters['rake'],
                                     time=parameters['origin_time'], sdr=sdr)
        else:
            source = self.get_source(la_s=parameters['la_s'], lo_s=parameters['lo_s'], depth=parameters['depth_s'],
                                     m_tp=parameters['m_tp'], m_rp=parameters['m_rp'], m_rt=parameters['m_rt'],
                                     m_pp=parameters['m_pp'], m_tt=parameters['m_tt'], m_rr=parameters['m_rr'],
                                     time=parameters['origin_time'], sdr=sdr)
        receiver = self.get_receiver()
        traces = self.db.get_seismograms(source=source, receiver=receiver, components=self.prior['components'],
                                         kind=self.prior['kind'])
        traces.interpolate(sampling_rate=2.0)
        traces.traces[0].data = np.float64(traces.traces[0].data)
        traces.traces[1].data = np.float64(traces.traces[1].data)
        traces.traces[2].data = np.float64(traces.traces[2].data)

        if noise_model == True:
            seismogram_noise.add_noise(traces, model=self.prior['noise_model'])
        seismogram = np.array([])
        for trace in traces.traces:
            seismogram = np.append(seismogram, trace)

        return seismogram, traces, source

    def get_P(self, epi, depth_m):
        model = TauPyModel(model=self.veloc_model)
        tt = model.get_travel_times(source_depth_in_km=depth_m / 1000, distance_in_degree=epi,
                                    phase_list=['P'])

        return tt[0].time

    def get_S(self, epi, depth_m):
        model = TauPyModel(model=self.veloc_model)
        tt = model.get_travel_times(source_depth_in_km=depth_m / 1000, distance_in_degree=epi,
                                    phase_list=['S'])

        return tt[0].time

    def get_receiver_time(self, epi, depth, origin_time):
        tt_P = self.get_P(epi, depth)  # Estimated P-wave arrival, based on the known velocity model
        tt_window = tt_P -5
        time_at_receiver = obspy.UTCDateTime(origin_time.timestamp + tt_window)
        return time_at_receiver

    def time_between_windows(self,epi,depth,dt):
        tt_P = self.get_P(epi, depth)  # Estimated P-wave arrival, based on the known velocity model
        tt_S = self.get_S(epi, depth)  # Estimated S-wave arrival, based on the known velocity model
        p_end = tt_P + 20
        s_start= tt_S -5
        tt_diff=s_start - p_end
        time_between_windows=tt_diff / dt
        return time_between_windows


    def get_window_obspy(self, seis_traces, epi,depth,or_time,npts):
        tt_P = self.get_P(epi, depth)  # Estimated P-wave arrival, based on the known velocity model
        tt_S = self.get_S(epi, depth)  # Estimated S-wave arrival, based on the known velocity model
        sec_per_sample = 1 / (seis_traces[0].meta.sampling_rate)

        total_stream = Stream()
        s_stream=Stream()
        p_stream=Stream()

        p_time = or_time.timestamp + tt_P
        s_time = or_time.timestamp + tt_S
        start_time_p = obspy.UTCDateTime(p_time - 5)  # -10 , + 44.2 --> PAPER: STAHLER & SIGLOCH
        end_time_p = obspy.UTCDateTime(p_time + 20)
        start_time_s = obspy.UTCDateTime(s_time - 15)
        end_time_s = obspy.UTCDateTime(s_time + 35)

        for i, trace in enumerate(seis_traces.traces):
            P_trace = Trace.slice(trace, start_time_p, end_time_p)
            # P_trace.detrend(type='demean')
            S_trace = Trace.slice(trace, start_time_s, end_time_s)
            # S_trace.detrend(type='demean')
            stream_add = P_trace.__add__(S_trace, fill_value=0, sanity_checks=True)
            zero_trace = Trace(np.zeros(npts),
                        header={"starttime": start_time_p, 'delta': trace.meta.delta, "station": trace.meta.station,
                                "network": trace.meta.network, "location": trace.meta.location,
                                "channel": trace.meta.channel})
            if 'T' in trace.meta.channel:
                total_trace = zero_trace.__add__(S_trace, method=0, interpolation_samples=0, fill_value=S_trace.data,
                                      sanity_checks=True)
                total_s_trace= total_trace.copy()
            else:
                total_trace=zero_trace.__add__(stream_add, method=0, interpolation_samples=0, fill_value=stream_add.data, sanity_checks=True)
                total_s_trace=zero_trace.__add__(S_trace, method=0, interpolation_samples=0, fill_value=S_trace.data, sanity_checks=True)
                total_p_trace=zero_trace.__add__(P_trace, method=0, interpolation_samples=0, fill_value=P_trace.data, sanity_checks=True)
                p_stream.append(total_p_trace)
            s_stream.append(total_s_trace)
            total_stream.append(total_trace)
            s_stream = self.BW_filter(s_stream)
            p_stream = self.BW_filter(p_stream)
            total_stream = self.BW_filter(total_stream)
        return total_stream,p_stream,s_stream,start_time_p,start_time_s
    def BW_filter(self,stream):
        stream.filter('highpass',freq=1.0/30.0)
        stream.filter('lowpass',freq=0.75)
        return stream

    def get_fft(self, traces, directory):

        for trace in traces:
            npts = trace.stats.npts
            t_tot = trace.stats.endtime.timestamp - trace.stats.starttime.timestamp
            df = trace.stats.sampling_rate  # npts / t_tot --> same
            dt = trace.stats.delta # t_tot / npts --> same
            fft = np.abs(np.fft.fft(trace.data))
            freq = np.arange(0, len(fft), 1) * df / len(fft)

            fig, ax = plt.subplots()
            ax.plot(freq, fft, label = "%s" % trace.id)
            ax.legend()
            plt.xlabel("Signal frequency [Hz]")
            plt.ylabel("Amplitude [Displacement]")
            plt.tight_layout()
            plt.savefig(directory + '/fft_channel_%s' %trace.stats.channel)
            plt.close()
            print("The fft graphs are saved in: %s" %directory)
            # dominant_freq = freq[np.argmax(fft[int(0.1*len(fft)):-int(0.1*len(fft))])]
            # return  dominant_freq

    def get_var_data(self,P_start_time,full_obs_stream):
        max_amp = 0
        noise_est = full_obs_stream.copy()
        for trace in noise_est:
            env = envelope(trace.data)
            trace.data = env
            trace.trim(starttime=trace.meta.starttime,endtime=P_start_time)
            max_amp += max(trace.data)
        return max_amp/(len(noise_est)*2)








