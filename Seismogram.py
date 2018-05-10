import instaseis
import numpy as np


class Seismogram:
    def __init__(self, PARAMETERS, db,sdr = False):
        self.par = PARAMETERS
        self.db = db
        self.receiver = instaseis.Receiver(latitude=self.par['la_r'], longitude=self.par['lo_r'],
                                           network=self.par['network'], station=self.par['station'])
        if sdr == True:
            self.source = instaseis.Source.from_strike_dip_rake(latitude=self.par['la_s'], longitude=self.par['lo_s'],
                                                            depth_in_m=self.par['depth_s'],
                                                            strike=self.par['strike'], dip=self.par['dip'],
                                                            rake=self.par['rake'], M0=self.par['M0'])
        else:
            self.source = instaseis.Source(latitude=self.par['la_s'], longitude=self.par['lo_s'],
                                           depth_in_m=self.par['depth_s'], m_tp=self.par['m_tp'], m_rp=self.par['m_rp'],
                                           m_rt=self.par['m_rt'], m_pp=self.par['m_pp'], m_tt=self.par['m_tt'],
                                           m_rr=self.par['m_rr'],origin_time=self.par['origin_time'])

    def get(self):
        traces = self.db.get_seismograms(source=self.source, receiver=self.receiver, components=self.par['components'],
                                         kind=self.par['kind'])
        seismogram = np.array([])
        for trace in traces.traces:
            seismogram = np.append(seismogram, trace)
        return seismogram, traces, self.source
