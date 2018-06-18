from obspy.taup import TauPyModel
from obspy.core.stream import Stream
from obspy.core.trace import Trace
import obspy
import numpy as np

class Source_code:
    def __init__(self, veloc_model):
        self.veloc_model = veloc_model

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

    def get_window_obspy(self, seis_traces, epi,depth,time,npts):
        tt_P = self.get_P(epi, depth)  # Estimated P-wave arrival, based on the known velocity model
        tt_S = self.get_S(epi, depth)  # Estimated S-wave arrival, based on the known velocity model
        sec_per_sample = 1 / (seis_traces[0].meta.sampling_rate)

        total_stream = Stream()
        s_stream=Stream()
        p_stream=Stream()
        p_time = time.timestamp + tt_P
        s_time = time.timestamp + tt_S
        start_time_p = obspy.UTCDateTime(p_time - 10)
        start_time_s = obspy.UTCDateTime(s_time - 10)
        end_time_p = obspy.UTCDateTime(p_time + 44.2)
        end_time_s = obspy.UTCDateTime(s_time + 44.2)

        for i, trace in enumerate(seis_traces.traces):
            P_trace = Trace.slice(trace, start_time_p, end_time_p)
            S_trace = Trace.slice(trace, start_time_s, end_time_s)
            stream_add = P_trace.__add__(S_trace, fill_value=0, sanity_checks=True)
            zero_trace = Trace(np.zeros(npts),
                        header={"starttime": start_time_p, 'delta': trace.meta.delta, "station": trace.meta.station,
                                "network": trace.meta.network, "location": trace.meta.location,
                                "channel": trace.meta.channel, "instaseis": trace.meta.instaseis})
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
            # TODO Get the Rayleigh and Love windows also!!
        return total_stream,p_stream,s_stream
