import instaseis
import numpy as np


class Seismogram:
    def __init__(self, PARAMETERS, db):
        self.par = PARAMETERS
        self.db = db

    def get_receiver(self):
        receiver = instaseis.Receiver(latitude=self.par['la_r'], longitude=self.par['lo_r'],
                                           network=self.par['network'], station=self.par['station'])
        return receiver

    def get_source(self, la_s, lo_s, depth, strike=None, dip=None, rake=None, m_tp=None, m_rp=None, m_rt=None,
                   m_pp=None, m_tt=None, m_rr=None, time=None, sdr=False):
        if sdr == True:
            source = instaseis.Source.from_strike_dip_rake(latitude=la_s, longitude=lo_s,
                                                                depth_in_m=depth,
                                                                strike=strike, dip=dip,
                                                                rake=rake, M0=self.par['M0'],
                                                                origin_time=time)
            return source
        else:
            source = instaseis.Source(latitude=la_s, longitude=lo_s,
                                           depth_in_m=depth, m_tp=m_tp, m_rp=m_rp,
                                           m_rt=m_rt, m_pp=m_pp, m_tt=m_tt,
                                           m_rr=m_rr, origin_time=time)
            return source

    def get_seis_manual(self, la_s, lo_s, depth, strike, dip, rake, time, sdr):
        source = self.get_source(la_s=la_s, lo_s=lo_s, depth=depth, strike=strike, dip=dip,rake= rake, time= time, sdr=sdr)
        receiver = self.get_receiver()
        traces = self.db.get_seismograms(source=source, receiver=receiver, components=self.par['components'],
                                         kind=self.par['kind'])
        seismogram = np.array([])
        for trace in traces.traces:
            seismogram = np.append(seismogram, trace)
        return seismogram, traces, source

    def get_seis_automatic(self, sdr=False):
        if sdr == True:
            source = self.get_source(la_s=self.par['la_s'], lo_s=self.par['lo_s'], depth=self.par['depth_s'],
                                     strike=self.par['strike'], dip=self.par['dip'], rake=self.par['rake'],
                                     time=self.par['origin_time'], sdr=sdr)
        else:
            source = self.get_source(la_s=self.par['la_s'], lo_s=self.par['lo_s'], depth=self.par['depth_s'],
                                     m_tp=self.par['m_tp'], m_rp=self.par['m_rp'], m_rt=self.par['m_rt'],
                                     m_pp=self.par['m_pp'], m_tt=self.par['m_tt'], m_rr=self.par['m_rr'],
                                     time=self.par['origin_time'], sdr=sdr)
        receiver = self.get_receiver()
        traces = self.db.get_seismograms(source=source, receiver=receiver, components=self.par['components'],
                                         kind=self.par['kind'])
        seismogram = np.array([])
        for trace in traces.traces:
            seismogram = np.append(seismogram, trace)
        return seismogram, traces, source
