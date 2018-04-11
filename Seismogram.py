import instaseis
import numpy as np

class Seismogram:
    def __init__(self, PARAMETERS, db):
        self.par = PARAMETERS
        self.db = db
        self.receiver = instaseis.Receiver(latitude=PARAMETERS['la_r'], longitude=PARAMETERS['lo_r'],
                                           network=PARAMETERS['network'], station=PARAMETERS['station'])
        self.source = instaseis.Source.from_strike_dip_rake(latitude=PARAMETERS['la_s'], longitude=PARAMETERS['lo_s'],
                                                            depth_in_m=PARAMETERS['depth_s'],
                                                            strike=PARAMETERS['strike'], dip=PARAMETERS['dip'],
                                                            rake=PARAMETERS['rake'], M0=PARAMETERS['M0'])

    def get(self):
        traces = self.db.get_seismograms(source=self.source, receiver=self.receiver, components=self.par['components'],
                                         kind=self.par['kind'])
        seismogram = np.array([])
        for trace in traces.traces:
            seismogram = np.append(seismogram, trace)
        return seismogram, traces, self.source

