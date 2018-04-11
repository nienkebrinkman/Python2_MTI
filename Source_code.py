from obspy.taup import TauPyModel
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
            G_P = G[sample_P_min:sample_P_max, :]
            G_S = G[sample_S_min:sample_S_max, :]
            G_new = np.vstack((G_P, G_S)) if i == 0 else np.vstack((G_new, G_P, G_S))
        return G_new, new_trace

    def def_phases(self):
        # mqs2019nspq
        a = 1
        phases = []
        phases.append(dict(starttime=lambda dist, depth: origin.time + dist / 2.8,
                           endtime=lambda dist, depth: origin.time + dist / 2.6,
                           comp='Z',
                           fmin=1. / 20.,
                           fmax=1. / 10.,
                           dt=5.0,
                           name='R1_10_20'))
        # phases.append(dict(starttime=lambda dist, depth: origin.time + get_P(depth, dist) - 5,
        #                   endtime=lambda dist, depth: origin.time + get_P(depth, dist) + 35,
        #                   comp='Z',
        #                   fmin=1./20.,
        #                   fmax=1./5.,
        #                   dt=2.0,
        #                   name='P_10_20'))
        phases.append(dict(starttime=lambda dist, depth: origin.time + dist / 2.8,
                           endtime=lambda dist, depth: origin.time + dist / 2.6,
                           comp='Z',
                           fmin=1. / 16.,
                           fmax=1. / 8.,
                           dt=4.0,
                           name='R1_08_16'))
        phases.append(dict(starttime=lambda dist, depth: origin.time + dist / 2.8,  # 2.7,
                           endtime=lambda dist, depth: origin.time + dist / 2.5,  # 2.5,
                           comp='Z',
                           fmin=1. / 32.,
                           fmax=1. / 16.,
                           dt=8.0,
                           name='R1_16_32'))
        phases.append(dict(starttime=lambda dist, depth: origin.time + dist / 2.8,  # /2.7,
                           endtime=lambda dist, depth: origin.time + dist / 2.4,  # /2.5,
                           comp='Z',
                           fmin=1. / 48.,
                           fmax=1. / 24.,
                           dt=12.0,
                           name='R1_24_48'))
        # phases.append(dict(starttime=lambda dist, depth: origin.time+(3389.5*2*np.pi - dist)/2.7,
        #                   endtime=lambda dist, depth: origin.time+(3389.5*2*np.pi - dist)/2.5,
        #                   comp='Z',
        #                   fmin=1./32.,
        #                   fmax=1./16.,
        #                   dt=8.0,
        #                   name='R2_16_32'))
        phases.append(dict(starttime=lambda dist, depth: origin.time + dist / 3.2,  # /3.15,
                           endtime=lambda dist, depth: origin.time + dist / 2.85,  # /2.95,
                           comp='T',
                           fmin=1. / 48.,
                           fmax=1. / 24.,
                           dt=5.0,
                           name='G1_24_48'))
        phases.append(dict(starttime=lambda dist, depth: origin.time + dist / 3.25,  # /3.2,
                           endtime=lambda dist, depth: origin.time + dist / 2.9,  # /2.95,
                           comp='T',
                           fmin=1. / 32.,
                           fmax=1. / 16.,
                           dt=8.0,
                           name='G1_16_32'))
        phases.append(dict(starttime=lambda dist, depth: origin.time + dist / 3.2,  # /3.15,
                           endtime=lambda dist, depth: origin.time + dist / 2.9,  # /2.95,
                           comp='T',
                           fmin=1. / 24.,
                           fmax=1. / 12.,
                           dt=6.0,
                           name='G1_12_24'))
        phases.append(dict(starttime=lambda dist, depth: origin.time + dist / 3.15,  # /3.1,
                           endtime=lambda dist, depth: origin.time + dist / 2.9,  # /3.00,
                           comp='T',
                           fmin=1. / 16.,
                           fmax=1. / 8.,
                           dt=4.0,
                           name='G1_08_16'))
        return phases


