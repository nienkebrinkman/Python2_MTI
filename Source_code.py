from obspy.taup import TauPyModel
import numpy as np
import matplotlib.pylab as plt

class Source_code:
    def __init__(self, veloc_model):
        self.veloc_model=veloc_model

    def get_P(self,epi,depth_m):
        model = TauPyModel(model=self.veloc_model)
        tt = model.get_travel_times(source_depth_in_km=depth_m/ 1000, distance_in_degree=epi,
                                    phase_list=['P'])

        return tt[0].time

    def get_S(self,epi,depth_m):
        model = TauPyModel(model=self.veloc_model)
        tt = model.get_travel_times(source_depth_in_km=depth_m/ 1000, distance_in_degree=epi,
                                    phase_list=['S'])
        return tt[0].time

    def get_windows(self,seis_traces, epi, depth):
        tt_P = self.get_P(epi,depth)  # Estimated P-wave arrival, based on the known velocity model
        tt_S = self.get_S(epi,depth)  # Estimated S-wave arrival, based on the known velocity model
        sec_per_sample = 1 / (seis_traces[0].meta.sampling_rate)

        seismogram = np.array([])
        # save_windows={}
        for i, trace in enumerate(seis_traces.traces):
            new_trace = np.zeros_like(trace)
            sample_P = int(tt_P / sec_per_sample)
            sample_S = int(tt_S / sec_per_sample)
            ## Define window for P-arrival:
            tt_diff = np.abs(sample_S - sample_P)
            sample_P_min = int(sample_P - (10/sec_per_sample))
            sample_P_max = int(sample_P + (44.2/sec_per_sample))
            P_diff=sample_P_max-sample_P_min
            new_trace[sample_P_min:sample_P_max]= trace[sample_P_min:sample_P_max]

            ## Define window for S-arrival:
            sample_S_min = int(sample_S - (10/sec_per_sample))
            sample_S_max = int(sample_S + (44.2/sec_per_sample))
            new_trace[sample_S_min:sample_S_max] = trace[sample_S_min:sample_S_max]
            seismogram = np.append(seismogram, new_trace)
        return seismogram

    def get_G(self,traces,G,epi,depth):
        tt_P = self.get_P(epi,depth)  # Estimated P-wave arrival, based on the known velocity model
        tt_S = self.get_S(epi,depth)  # Estimated S-wave arrival, based on the known velocity model
        ## Get P- & S- windows also from the G matrix:
        G_new = np.zeros_like(G)
        sec_per_sample = 1 / (traces[0].meta.sampling_rate)

        for i,trace in enumerate(traces.traces):

            sample_P = int(tt_P / sec_per_sample)
            sample_P_min = int(sample_P - (10/sec_per_sample))
            sample_P_max = int(sample_P + (44.2/sec_per_sample))

            sample_S = int(tt_S / sec_per_sample)
            sample_S_min = int(sample_S - (10/sec_per_sample))
            sample_S_max = int(sample_S + (44.2/sec_per_sample))

            G_new[sample_P_min + len(trace) * i:sample_P_max + len(trace) * i, :] = G[sample_P_min + len(
                trace) * i:sample_P_max + len(trace) * i, :]
            G_new[sample_S_min + len(trace) * i:sample_S_max + len(trace) * i, :] = G[sample_S_min + len(
                trace) * i:sample_S_max + len(trace) * i, :]
        return G_new




