import instaseis
import seismogram_noise
from obspy.taup import TauPyModel
import obspy
import numpy as np
from obspy.core.stream import Stream
from obspy.core.trace import Trace
import matplotlib.pylab as plt

class Create_observed:
    def __init__(self, PRIOR,PARAMETERS, db):
        self.par = PARAMETERS
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
                                                           origin_time=time)
            return source
        else:
            source = instaseis.Source(latitude=la_s, longitude=lo_s,
                                           depth_in_m=depth, m_tp=m_tp, m_rp=m_rp,
                                           m_rt=m_rt, m_pp=m_pp, m_tt=m_tt,
                                           m_rr=m_rr, origin_time=time)
            return source

    def get_seis_automatic(self, prior, noise_model=False, sdr=False):
        self.prior = prior
        if sdr == True:
            source = self.get_source(la_s=self.par['la_s'], lo_s=self.par['lo_s'], depth=self.par['depth_s'],
                                     strike=self.par['strike'], dip=self.par['dip'], rake=self.par['rake'],
                                     time=self.par['origin_time'], sdr=sdr)
        else:
            source = self.get_source(la_s=self.par['la_s'], lo_s=self.par['lo_s'], depth=self.par['depth_s'],
                                     m_tp=self.par['m_tp'], m_rp=self.par['m_rp'], m_rt=self.par['m_rt'],
                                     m_pp=self.par['m_pp'], m_tt=self.par['m_tt'], m_rr=self.par['m_rr'],
                                     time=self.par['origin_time'], sdr=sdr)
        receiver = self.get_receiver()
        traces = self.db.get_seismograms(source=source, receiver=receiver, components=self.prior['components'],
                                         kind=self.prior['kind'])
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

    def get_receiver_time(self, epi, depth, d_obs_traces):
        tt_P = self.get_P(epi, depth)  # Estimated P-wave arrival, based on the known velocity model
        tt_window = tt_P -10
        time_at_receiver = obspy.UTCDateTime(self.par['origin_time'].timestamp + tt_window)
        # TODO plot the reference time on the actual d_obs
        # plt.plot(d_obs_traces[0])
        # plt.plot()
        return time_at_receiver

    def time_between_windows(self,epi,depth,dt):
        tt_P = self.get_P(epi, depth)  # Estimated P-wave arrival, based on the known velocity model
        tt_S = self.get_S(epi, depth)  # Estimated S-wave arrival, based on the known velocity model
        p_end = tt_P + 44.2
        s_start= tt_S -10
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
        for i, trace in enumerate(seis_traces.traces):
            p_time = or_time.timestamp + tt_P
            s_time=or_time.timestamp+tt_S
            start_time_p = obspy.UTCDateTime(p_time - 10) # -10 , + 44.2 --> PAPER: STAHLER & SIGLOCH
            end_time_p = obspy.UTCDateTime(p_time + 44.2)
            start_time_s = obspy.UTCDateTime(s_time - 10)
            end_time_s = obspy.UTCDateTime(s_time + 44.2)

            P_trace = Trace.slice(trace, start_time_p, end_time_p)
            S_trace = Trace.slice(trace, start_time_s, end_time_s)
            stream_add = P_trace.__add__(S_trace, fill_value=0, sanity_checks=True)
            zero_trace = Trace(np.zeros(npts),
                        header={"starttime": start_time_p, 'delta': trace.meta.delta, "station": trace.meta.station,
                                "network": trace.meta.network, "location": trace.meta.location,
                                "channel": trace.meta.channel, "instaseis": trace.meta.instaseis})
            if trace.meta.channel == u'LXT':
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
            # TODO Get the Rayleigh and Love windows also!!
        return total_stream,p_stream,s_stream



