import numpy as np
import instaseis
import obspy
from obspy.core.trace import Trace
from obspy.core.stream import Stream
from obspy.geodetics.base import gps2dist_azimuth

from Get_Parameters import Get_Paramters
from Source_code import Source_code
from Surface_waves import Surface_waves



class Green_functions:
    ## Obtain Greens-function from Instaseis - [MxN] : M-number of rows, N-number of columns
    def __init__(self, PRIOR, db):
        self.par = PRIOR
        self.db = db
        self.window = Source_code(self.par['VELOC_taup'])

    # def get_old(self,epi,depth,time):
    #     gf = self.db.get_greens_function(epicentral_distance_in_degree=epi,
    #                                      source_depth_in_m=depth, origin_time=time,
    #                                      kind=self.par['kind'], kernelwidth=self.par['kernelwidth'],
    #                                      definition=self.par['definition'])
    #     az = self.par['az']
    #     raz = np.deg2rad(az)
    #
    #     tss = gf.traces[0].data
    #     zss = gf.traces[1].data
    #     rss = gf.traces[2].data
    #     tds = gf.traces[3].data
    #     zds = gf.traces[4].data
    #     rds = gf.traces[5].data
    #     zdd = gf.traces[6].data
    #     rdd = gf.traces[7].data
    #     zep = gf.traces[8].data
    #     rep = gf.traces[9].data
    #
    #     G_z = gf.traces[0].meta['npts']
    #     G_r = gf.traces[0].meta['npts'] * 2
    #     G_t = gf.traces[0].meta['npts'] * 3
    #     G = np.ones((G_t, 6))
    #     G[0:G_z, 0] = 0.5 * zss *np.cos(2*raz) - zdd *(1.0/6.0) + zep * (1.0/3.0)
    #     G[0:G_z, 1] = -0.5 * zss *np.cos(2*raz) - zdd *(1.0/6.0) + zep * (1.0/3.0)
    #     G[0:G_z, 2] = zdd * (1.0/3.0) + zep * (1.0/3.0)
    #     G[0:G_z, 3] = zss *np.sin(2*raz)
    #     G[0:G_z, 4] = zds * np.cos(raz)
    #     G[0:G_z, 5] = zds * np.sin(raz)
    #
    #     G[G_z:G_r, 0] = 0.5 * rss * np.cos(2*raz) - rdd * (1.0/6.0) + rep * (1.0/3.0)
    #     G[G_z:G_r, 1] = -0.5 * rss * np.cos(2*raz) - rdd * (1.0/6.0) + rep * (1.0/3.0)
    #     G[G_z:G_r, 2] = rdd * (1.0/3.0) + rep * (1.0/3.0)
    #     G[G_z:G_r, 3] = rss * np.sin(2*raz)
    #     G[G_z:G_r, 4] = rds * np.cos(raz)
    #     G[G_z:G_r, 5] = rds * np.sin(raz)
    #
    #     G[G_r:G_t, 0] = -tss * (0.5) * np.sin(2 * np.deg2rad(self.par['az']))
    #     G[G_r:G_t, 1] = tss * (0.5) * np.sin(2 * np.deg2rad(self.par['az']))
    #     G[G_r:G_t, 2] =   0
    #     G[G_r:G_t, 3] = tss * np.cos(2 * np.deg2rad(self.par['az']))
    #     G[G_r:G_t, 4] = tds * np.sin(2 * np.deg2rad(self.par['az']))
    #     G[G_r:G_t, 5] = -tds * np.cos(2 * np.deg2rad(self.par['az']))
    #     return G

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
        G[G_r:G_t, 3] = tds * np.sin(np.deg2rad(self.par['az']))
        G[G_r:G_t, 4] = -tds * np.cos(np.deg2rad(self.par['az']))
        return G

    def get_bw(self, time_at_rec, epi, depth, npts_trace):
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
        G_t[:, 3] = tds * np.sin(np.deg2rad(self.par['az']))
        G_t[:, 4] = -tds * np.cos(np.deg2rad(self.par['az']))


        G_tot = np.vstack((np.vstack((G_z,G_r)),G_t))

        return G_tot, G_z, G_r, G_t

    def get_sw(self, time_at_rec, epi, depth, la_s,lo_s ,npts_trace):
        gf = self.db.get_greens_function(epicentral_distance_in_degree=epi, source_depth_in_m=depth,
                                         origin_time=time_at_rec, kind=self.par['kind'],
                                         kernelwidth=self.par['kernelwidth'],
                                         definition=self.par['definition'])

        dist, az, baz = gps2dist_azimuth(lat1=la_s,
                                         lon1=lo_s,
                                         lat2=self.par['la_r'],
                                         lon2=self.par['lo_r'], a=self.par['radius'], f=0)


        Surface = Surface_waves(self.par)
        Phases_R = Surface.get_R_phases(time_at_rec)
        Phases_L = Surface.get_L_phases(time_at_rec)


        for j,R in enumerate(Phases_R):
            R_format_gf = Stream()
            run = gf.copy()
            for i in range(len(gf.traces)):
                if 'Z' not in gf.traces[i].id:
                    continue
                else:
                    R_trim=run.traces[i].trim(starttime=R['starttime'](dist, depth), endtime=R['endtime'](dist, depth))

                    zero_trace = Trace(np.zeros(npts_trace),
                                       header={"starttime": R['starttime'](dist, depth), 'delta': gf.traces[0].meta.delta,
                                               'definition': self.par['definition'], 'kernelwidth': self.par['kernelwidth'],
                                               'kind': self.par['kind'], "instaseis": gf.traces[0].meta.instaseis,
                                               'channel': gf.traces[i].id})
                    R_Trace = Trace(R_trim.data,
                                      header={"starttime": R['starttime'](dist, depth), 'delta': gf.traces[0].meta.delta,
                                              'definition': self.par['definition'], 'kernelwidth': self.par['kernelwidth'],
                                              'kind': self.par['kind'], "instaseis": gf.traces[0].meta.instaseis,
                                              'channel': gf.traces[i].id})
                    filled_trace = zero_trace.__add__(R_Trace, method=0, interpolation_samples=0,
                                                      fill_value=R_Trace.data,
                                                      sanity_checks=False)
                    R_format_gf.append(filled_trace)



            zss = R_format_gf.traces[0].data
            zds = R_format_gf.traces[1].data
            zdd = R_format_gf.traces[2].data
            zep = R_format_gf.traces[3].data

            G_R_Z = np.ones((npts_trace, 5))
            G_R_Z[:, 0] = zss * (0.5) * np.cos(2 * np.deg2rad(self.par['az'])) - zdd * 0.5
            G_R_Z[:, 1] = - zdd * 0.5 - zss * (0.5) * np.cos(2 * np.deg2rad(self.par['az']))
            # R_Zz G[0:G_z, 1] =  zdd * (1/3) + zep * (1/3)
            G_R_Z[:, 2] = zss * np.sin(2 * np.deg2rad(self.par['az']))
            G_R_Z[:, 3] = -zds * np.cos(np.deg2rad(self.par['az']))
            G_R_Z[:, 4] = -zds * np.sin(np.deg2rad(self.par['az']))


            R_new = np.vstack((G_R_Z)) if j == 0 else np.vstack((R_new, G_R_Z))


        for j, L in enumerate(Phases_L):
            L_format_gf = Stream()
            run = gf.copy()
            for i in range(len(gf.traces)):
                if 'T' not in gf.traces[i].id:
                    continue
                else:
                    L_trim = run.traces[i].trim(starttime=L['starttime'](dist, depth),
                                               endtime=L['endtime'](dist, depth))

                    zero_trace = Trace(np.zeros(npts_trace),
                                       header={"starttime": L['starttime'](dist, depth),
                                               'delta': gf.traces[0].meta.delta,
                                               'definition': self.par['definition'],
                                               'kernelwidth': self.par['kernelwidth'],
                                               'kind': self.par['kind'], "instaseis": gf.traces[0].meta.instaseis,
                                               'channel': gf.traces[i].id})
                    L_Trace = Trace(L_trim.data,
                                    header={"starttime": L['starttime'](dist, depth),
                                            'delta': gf.traces[0].meta.delta,
                                            'definition': self.par['definition'],
                                            'kernelwidth': self.par['kernelwidth'],
                                            'kind': self.par['kind'], "instaseis": gf.traces[0].meta.instaseis,
                                            'channel': gf.traces[i].id})
                    filled_trace = zero_trace.__add__(L_Trace, method=0, interpolation_samples=0,
                                                      fill_value=L_Trace.data,
                                                      sanity_checks=False)
                    L_format_gf.append(filled_trace)

            tss = L_format_gf.traces[0].data
            tds = L_format_gf.traces[1].data

            G_L_T = np.ones((npts_trace, 5))

            G_L_T[:, 0] = -tss * (0.5) * np.sin(2 * np.deg2rad(self.par['az']))
            G_L_T[:, 1] = tss * (0.5) * np.sin(2 * np.deg2rad(self.par['az']))
            # L_Tt G[G_r:G_t, 1] =   0
            G_L_T[:, 2] = tss * np.cos(2 * np.deg2rad(self.par['az']))
            G_L_T[:, 3] = tds * np.sin(np.deg2rad(self.par['az']))
            G_L_T[:, 4] = -tds * np.cos(np.deg2rad(self.par['az']))

            L_new = np.vstack((G_L_T)) if j == 0 else np.vstack((L_new, G_L_T))
        G_R_L = np.vstack((R_new,L_new))
        return G_R_L, R_new ,L_new