## NOT FINISHED YET
import numpy as np

class Neighborhood_algorithm:
    def __init__(self, PARAMETERS, sampler, db, data):
        self.db = db
        self.par = PARAMETERS
        self.sampler = sampler
        self.d_obs = data

        epi = np.linspace(self.sampler['epi']['range_min'], self.sampler['epi']['range_max'],
                          self.sampler['epi']['steps'])
        azimuth = np.linspace(self.sampler['azimuth']['range_min'], self.sampler['azimuth']['range_max'],
                              self.sampler['azimuth']['steps'])
        depth = np.linspace(self.sampler['depth']['range_min'], self.sampler['depth']['range_max'],
                            self.sampler['depth']['steps'])

        # Time sampler: !!!!!!!!!!!!!!!!NOT CORRECT YET!!!!!!!!!!
        # time_obs = self.par['origin_time'] # Observed time at station
        # time_sample_min = obspy.UTCDateTime(time_obs.year, time_obs.month, time_obs.day,self.sampler['time_range'] , time_obs.min, time_obs.sec)
        # time = np.linspace(time_sample_min,time_obs,step)

        self.dict = {'epi': epi,
                     'azimuth': azimuth,
                     'depth': depth}

    def make_grid(self):
        # Call the keys within the dictionary:
        self.keys = self.dict.keys()

        # Call the values within the dictionary:
        self.val = self.dict.values()

        # NOT FINISHED YET

