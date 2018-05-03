from obspy.taup import TauPyModel
import numpy as np

class Source_code:
    def __init__(self, PARAMETERS, db):
        self.par = PARAMETERS
        self.db = db

    def get_P(self):
        model = TauPyModel(model=self.par['VELOC_taup'])
        tt = model.get_travel_times(source_depth_in_km=self.par['depth_s'] / 1000, distance_in_degree=self.par['epi'],
                                    phase_list=['P'], receiver_depth_in_km=0.0)
        return tt[0].time

    def get_S(self):
        model = TauPyModel(model=self.par['VELOC_taup'])
        tt = model.get_travel_times(source_depth_in_km=self.par['depth_s'] / 1000, distance_in_degree=self.par['epi'],
                                    phase_list=['S'], receiver_depth_in_km=0.0)
        return tt[0].time

    def get_windows(self, traces, G):
        tt_P = self.get_P()  # Estimated P-wave arrival, based on the known velocity model
        tt_S = self.get_S()  # Estimated S-wave arrival, based on the known velocity model
        sec_per_sample = 1 / (traces[0].meta.sampling_rate)

        new_trace = np.array([])
        for i, trace in enumerate(traces.traces):
            sample_P = int(tt_P / sec_per_sample)
            sample_S = int(tt_S / sec_per_sample)
            ## Define window for P-arrival:
            tt_diff = np.abs(sample_S - sample_P)
            sample_P_min = int(sample_P - tt_diff * 0.5)
            sample_P_max = int(sample_P + tt_diff * 0.5 + 1)
            p_array = trace[sample_P_min:sample_P_max]
            new_trace = np.append(new_trace, p_array)

            ## Define window for S-arrival:
            sample_S_min = int(sample_S - tt_diff * 0.5)
            sample_S_max = int(sample_S + tt_diff * 0.5 + 1)
            s_array = trace[sample_S_min:sample_S_max]
            new_trace = np.append(new_trace, s_array)

            ## Get P- & S- windows also from the G matrix:
            G_P = G[sample_P_min+len(trace)*i:sample_P_max+ len(trace)*i, :]
            G_S = G[sample_S_min+ len(trace)*i:sample_S_max+ len(trace)*i, :]
            G_new = np.vstack((G_P, G_S)) if i == 0 else np.vstack((G_new, G_P, G_S))
        return G_new, new_trace


