from obspy.taup import TauPyModel
from obspy.core.stream import Stream
from obspy.core.trace import Trace
import obspy
import numpy as np

class Source_code:
    def __init__(self, veloc_model_taup):
        self.veloc_model = veloc_model_taup

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
        start_time_p = obspy.UTCDateTime(p_time - 5)
        start_time_s = obspy.UTCDateTime(s_time - 5)
        end_time_p = obspy.UTCDateTime(p_time + 20)
        end_time_s = obspy.UTCDateTime(s_time + 20)

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
        return total_stream,p_stream,s_stream

    def stack_BW_SW_Streams(self,traces_BW,traces_RW,traces_LW):
        stack_stream = traces_BW + traces_RW + traces_LW
        return stack_stream


    def stack_traces(self,stream):
        stacked_traces=np.array([])
        for trace in stream.traces:
            stacked_traces = np.append(stacked_traces,trace.data)
        return stacked_traces


    def split_traces(self,d_syn, traces_obs,time_at_receiver):
        Stream_split= Stream()
        for i,trace in enumerate(traces_obs.traces):
            new_trace = Trace(d_syn[i*len(trace):i*len(trace)+len(trace)],
                               header={"starttime": time_at_receiver, 'delta': trace.meta.delta, "station": trace.meta.station,
                                       "network":trace.meta.network, "location": trace.meta.location,
                                       "channel": trace.meta.channel, "instaseis": trace.meta.instaseis})
            Stream_split.append(new_trace)

        return Stream_split

    def split_BW_SW(self,BW_SW_stream, epi, depth, time_at_receiver, npts):
        BW_stream = Stream()
        R_stream = Stream()
        L_stream = Stream()
        for i in BW_SW_stream:
            if 'X' in i.id:
                BW_stream.append(i)

            elif 'R1' in i.id:
                R_stream.append(i)

            elif 'G1' in i.id:
                L_stream.append(i)

        P_S_syn, P_syn, S_syn = self.get_window_split_syn(BW_stream, epi, depth, time_at_receiver, npts)
        return P_S_syn,P_syn,S_syn,R_stream,L_stream



    def get_window_split_syn(self, splitted_syn,epi,depth, time_at_receiver,npts):
        tt_P = self.get_P(epi, depth)  # Estimated P-wave arrival, based on the known velocity model
        tt_S = self.get_S(epi, depth)  # Estimated S-wave arrival, based on the known velocity model

        diff = tt_S - tt_P

        P_start = time_at_receiver
        P_end = obspy.UTCDateTime(P_start+5+20)
        S_start = obspy.UTCDateTime(time_at_receiver.timestamp + diff)
        S_end = obspy.UTCDateTime(S_start+5+20)

        p_stream = Stream()
        s_stream = Stream()
        total_stream = Stream()

        for i, trace in enumerate(splitted_syn.traces):
            P_trace = Trace.slice(trace, P_start, P_end)
            S_trace = Trace.slice(trace, S_start, S_end)
            stream_add = P_trace.__add__(S_trace, fill_value=0, sanity_checks=True)
            zero_trace = Trace(np.zeros(npts),
                        header={"starttime": P_start, 'delta': trace.meta.delta, "station": trace.meta.station,
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
        return total_stream,p_stream,s_stream

