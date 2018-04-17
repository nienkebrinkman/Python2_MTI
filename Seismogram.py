import instaseis
import numpy as np


class Seismogram:
    def __init__(self, PARAMETERS, db):
        self.par = PARAMETERS
        self.db = db
        self.receiver = instaseis.Receiver(latitude=self.par['la_r'], longitude=self.par['lo_r'],
                                           network=self.par['network'], station=self.par['station'])
        self.source = instaseis.Source.from_strike_dip_rake(latitude=self.par['la_s'], longitude=self.par['lo_s'],
                                                            depth_in_m=self.par['depth_s'],
                                                            strike=self.par['strike'], dip=self.par['dip'],
                                                            rake=self.par['rake'], M0=self.par['M0'])

    def get(self):

        traces = self.db.get_seismograms(source=self.source, receiver=self.receiver, components=self.par['components'],
                                         kind=self.par['kind'])
        seismogram = np.array([])
        for trace in traces.traces:
            seismogram = np.append(seismogram, trace)
        return seismogram, traces, self.source
