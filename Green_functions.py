import numpy as np
import instaseis
import obspy
from obspy.core.trace import Trace
from obspy.core.stream import Stream

from Get_Parameters import Get_Paramters
from Source_code import Source_code
#
#
# def main():
#     get_parameters = Get_Paramters()
#     PARAMETERS = get_parameters.get_unkown()
#     PRIOR = get_parameters.get_prior()
#     PRIOR['az'] = 12.0064880807
#
#     db = instaseis.open_db(PRIOR['VELOC'])
#
#     green = Green_functions(PRIOR, db)
#     G = green.get(PARAMETERS['origin_time'], PARAMETERS['epi'], PARAMETERS['depth_s'], 300)


class Green_functions:
    ## Obtain Greens-function from Instaseis - [MxN] : M-number of rows, N-number of columns
    def __init__(self, PRIOR, db):
        self.par = PRIOR
        self.db = db
        self.window = Source_code(self.par['VELOC_taup'])

    def get_old(self,epi,depth,time):
        gf = self.db.get_greens_function(epicentral_distance_in_degree=epi,
                                         source_depth_in_m=depth, origin_time=time,
                                         kind=self.par['kind'], kernelwidth=self.par['kernelwidth'],
                                         definition=self.par['definition'])
        tss = gf.traces[0].data
        zss = gf.traces[1].data
        rss = gf.traces[2].data
        tds = gf.traces[3].data
        zds = gf.traces[4].data
        rds = gf.traces[5].data
        zdd = gf.traces[6].data
        rdd = gf.traces[7].data
        zep = gf.traces[8].data
        rep = gf.traces[9].data

        G_z = gf.traces[0].meta['npts']
        G_r = gf.traces[0].meta['npts'] * 2
        G_t = gf.traces[0].meta['npts'] * 3
        G = np.ones((G_t, 5))
        G[0:G_z, 0] = zss * (0.5) * np.cos(2 * np.deg2rad(self.par['az'])) - zdd * 0.5
        G[0:G_z, 1] = - zdd * 0.5 - zss * (0.5) * np.cos(2 * np.deg2rad(self.par['az']))
        # G[0:G_z, 1] =  zdd * (1/3) + zep * (1/3)
        G[0:G_z, 2] = zss * np.sin(2 * np.deg2rad(self.par['az']))
        G[0:G_z, 3] = -zds * np.cos(np.deg2rad(self.par['az']))
        G[0:G_z, 4] = -zds * np.sin(np.deg2rad(self.par['az']))

        G[G_z:G_r, 0] = rss * (0.5) * np.cos(2 * np.deg2rad(self.par['az'])) - rdd * 0.5
        G[G_z:G_r, 1] = -0.5 * rdd - rss * (0.5) * np.cos(2 * np.deg2rad(self.par['az']))
        # G[G_z:G_r, 1] =  rdd * (1/3) + rep * (1/3)
        G[G_z:G_r, 2] = rss * np.sin(2 * np.deg2rad(self.par['az']))
        G[G_z:G_r, 3] = -rds * np.cos(np.deg2rad(self.par['az']))
        G[G_z:G_r, 4] = -rds * np.sin(np.deg2rad(self.par['az']))

        G[G_r:G_t, 0] = -tss * (0.5) * np.sin(2 * np.deg2rad(self.par['az']))
        G[G_r:G_t, 1] = tss * (0.5) * np.sin(2 * np.deg2rad(self.par['az']))
        # G[G_r:G_t, 1] =   0
        G[G_r:G_t, 2] = tss * np.cos(2 * np.deg2rad(self.par['az']))
        G[G_r:G_t, 3] = tds * np.sin(2 * np.deg2rad(self.par['az']))
        G[G_r:G_t, 4] = -tds * np.cos(2 * np.deg2rad(self.par['az']))
        return G

    def get(self, time_at_rec, epi, depth, npts_trace):
        gf = self.db.get_greens_function(epicentral_distance_in_degree=epi, source_depth_in_m=depth,
                                         origin_time=time_at_rec, kind=self.par['kind'],
                                         kernelwidth=self.par['kernelwidth'],
                                         definition=self.par['definition'])
        tt_P = self.window.get_P(epi,depth)
        tt_S = self.window.get_S(epi,depth)
        p_time = time_at_rec.timestamp + tt_P
        s_time = time_at_rec.timestamp + tt_S
        start_time_p = obspy.UTCDateTime(p_time - 10)
        start_time_s = obspy.UTCDateTime(s_time - 10)
        end_time_p = obspy.UTCDateTime(p_time + 44.2)
        end_time_s = obspy.UTCDateTime(s_time + 44.2)

        format_gf = Stream()
        for i in range(len(gf.traces)):
            if i == 0 or i == 3:
                stream_add = Trace.slice(gf.traces[i], start_time_s, end_time_s)
            else:
                P_trace = Trace.slice(gf.traces[i], start_time_p, end_time_p)
                S_trace = Trace.slice(gf.traces[i], start_time_s, end_time_s)
                stream_add = P_trace.__add__(S_trace, fill_value=0, sanity_checks=True)
            zero_trace = Trace(np.zeros(npts_trace),
                               header={"starttime": start_time_p, 'delta': gf.traces[0].meta.delta,
                                       'definition': self.par['definition'], 'kernelwidth': self.par['kernelwidth'],
                                       'kind': self.par['kind'], "instaseis": gf.traces[0].meta.instaseis,'channel':gf.traces[i].id})
            filled_trace=zero_trace.__add__(stream_add, method=0, interpolation_samples=0, fill_value=stream_add.data,
                               sanity_checks=False)
            format_gf.append(filled_trace)

        tss = format_gf.traces[0].data
        zss = format_gf.traces[1].data
        rss = format_gf.traces[2].data
        tds = format_gf.traces[3].data
        zds = format_gf.traces[4].data
        rds = format_gf.traces[5].data
        zdd = format_gf.traces[6].data
        rdd = format_gf.traces[7].data
        zep = format_gf.traces[8].data
        rep = format_gf.traces[9].data

        G_z = np.ones((npts_trace, 5))
        G_r = np.ones((npts_trace, 5))
        G_t = np.ones((npts_trace, 5))

        G_z[:, 0] = zss * (0.5) * np.cos(2 * np.deg2rad(self.par['az'])) - zdd * 0.5
        G_z[:, 1] = - zdd * 0.5 - zss * (0.5) * np.cos(2 * np.deg2rad(self.par['az']))
        # _z G[0:G_z, 1] =  zdd * (1/3) + zep * (1/3)
        G_z[:, 2] = zss * np.sin(2 * np.deg2rad(self.par['az']))
        G_z[:, 3] = -zds * np.cos(np.deg2rad(self.par['az']))
        G_z[:, 4] = -zds * np.sin(np.deg2rad(self.par['az']))

        G_r[:, 0] = rss * (0.5) * np.cos(2 * np.deg2rad(self.par['az'])) - rdd * 0.5
        G_r[:, 1] = -0.5 * rdd - rss * (0.5) * np.cos(2 * np.deg2rad(self.par['az']))
        # _r G[G_z:G_r, 1] =  rdd * (1/3) + rep * (1/3)
        G_r[:, 2] = rss * np.sin(2 * np.deg2rad(self.par['az']))
        G_r[:, 3] = -rds * np.cos(np.deg2rad(self.par['az']))
        G_r[:, 4] = -rds * np.sin(np.deg2rad(self.par['az']))

        G_t[:, 0] = -tss * (0.5) * np.sin(2 * np.deg2rad(self.par['az']))
        G_t[:, 1] = tss * (0.5) * np.sin(2 * np.deg2rad(self.par['az']))
        # _t G[G_r:G_t, 1] =   0
        G_t[:, 2] = tss * np.cos(2 * np.deg2rad(self.par['az']))
        G_t[:, 3] = tds * np.sin(2 * np.deg2rad(self.par['az']))
        G_t[:, 4] = -tds * np.cos(2 * np.deg2rad(self.par['az']))


        G_tot = np.vstack((np.vstack((G_z,G_r)),G_z))

        return G_tot, G_z, G_r, G_t

