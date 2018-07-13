import instaseis
import numpy as np
from obspy.core.stream import Stream

class Seismogram:
    def __init__(self, PRIOR, db):
        self.prior = PRIOR
        self.db = db

    def get_receiver(self):
        receiver = instaseis.Receiver(latitude=self.prior['la_r'], longitude=self.prior['lo_r'],
                                      network=self.prior['network'], station=self.prior['station'])
        return receiver

    def get_source(self, la_s, lo_s, depth, strike=None, dip=None, rake=None, m_tp=None, m_rp=None, m_rt=None,
                   m_pp=None, m_tt=None, m_rr=None, time=None, sdr=False):
        if sdr == True:
            source = instaseis.Source.from_strike_dip_rake(latitude=la_s, longitude=lo_s,
                                                           depth_in_m=depth,
                                                           strike=strike, dip=dip,
                                                           rake=rake, M0=self.prior['M0'],
                                                           origin_time=time,dt=2.0)
            return source
        else:
            source = instaseis.Source(latitude=la_s, longitude=lo_s,
                                           depth_in_m=depth, m_tp=m_tp, m_rp=m_rp,
                                           m_rt=m_rt, m_pp=m_pp, m_tt=m_tt,
                                           m_rr=m_rr, origin_time=time)
            return source

    def get_seis_manual(self, la_s, lo_s, depth, strike, dip,   rake, time, sdr):
        source = self.get_source(la_s=la_s, lo_s=lo_s, depth=depth, strike=strike, dip=dip,rake= rake, time= time, sdr=sdr)
        receiver = self.get_receiver()
        traces = self.db.get_seismograms(source=source, receiver=receiver, components=self.prior['components'],
                                         kind=self.prior['kind'])
        traces.interpolate(sampling_rate = 2.0)
        traces.traces[0].data = np.float64(traces.traces[0].data)
        traces.traces[1].data = np.float64(traces.traces[1].data)
        traces.traces[2].data = np.float64(traces.traces[2].data)
        seismogram = np.array([])
        for trace in traces.traces:
            seismogram = np.append(seismogram, trace)
        return seismogram, traces, source


