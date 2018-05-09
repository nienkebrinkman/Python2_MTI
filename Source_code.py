from obspy.taup import TauPyModel
import numpy as np

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


    def get_windows(self, traces, G, epi,depth_m):
        tt_P = self.get_P(epi,depth_m)  # Estimated P-wave arrival, based on the known velocity model
        tt_S = self.get_S(epi,depth_m)  # Estimated S-wave arrival, based on the known velocity model
        sec_per_sample = 1 / (traces[0].meta.sampling_rate)

        new_trace = np.array([])
        save_windows={}
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

            save_windows.update({'%i'%i: {'P_min':sample_P_min,'P_max':sample_P_max, 'P_len':len(p_array),'S_min':sample_S_min,'S_max':sample_S_max,'S_len':len(s_array)}})

            ## Get P- & S- windows also from the G matrix:
            G_P = G[sample_P_min+len(trace)*i:sample_P_max+ len(trace)*i, :]
            G_S = G[sample_S_min+ len(trace)*i:sample_S_max+ len(trace)*i, :]
            G_new = np.vstack((G_P, G_S)) if i == 0 else np.vstack((G_new, G_P, G_S))
        return G_new, new_trace,save_windows


