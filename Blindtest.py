# Moment tensor inversion for a Blindtest dataset

import obspy
import instaseis
import os
import numpy as np
from obspy.signal.filter import envelope
from mqscatalog import filter_by_location_quality, get_phase_picks
from obspy.core.event.event import Event
from obspy.geodetics.base import gps2dist_azimuth
from obspy.geodetics import kilometer2degrees
from obspy.geodetics import degrees2kilometers
from obspy.core.stream import Stream
from obspy.core.trace import Trace
import matplotlib.pylab as plt


class Blindtest:
    def get_events(self,filepath_catalog):
        catalog = obspy.read_events(filepath_catalog)
        return catalog.events

    def get_qualityA_event(self,filepath_catalog):
        cat =obspy.read_events(filepath_catalog)
        qualityA_catalog = filter_by_location_quality(catalog=cat, quality='A')
        return qualityA_catalog.events

    def get_pref_origin(self,event):
        source = Event.preferred_origin(event)
        depth = source.depth
        la_s = source.latitude
        lo_s = source.longitude
        time = source.time
        return time, depth, la_s, lo_s

    def get_pref_scalarmoment(self,event):
        magnitude = Event.preferred_magnitude(event)
        Mw = magnitude.mag
        M = self.Magnitude2Scalarmoment(Mw)
        return M

    def Magnitude2Scalarmoment(self,Mw):
        M=10**(9.1 + Mw *(3.0/2.0))
        return M
    def Scalarmoment2Magnitude(self,M0):
        Mw = 2.0 / 3.0 * (np.log10(M0) - 9.1)
        return Mw

    def pick_sw(self,stream,pick_info,epi,prior,npts, directory,plot_modus=False):
        if plot_modus == True:
            dir_SW = directory + '/Blind_rayleigh'
            if not os.path.exists(dir_SW):
                os.makedirs(dir_SW)
        Rayleigh_st = Stream()
        Love_st = Stream()

        dist = degrees2kilometers(epi,prior['radius'])
        phase = 0
        for pick in pick_info:
            if pick['phase_name'] == 'R1':
                if plot_modus == True:
                    dir_phases = dir_SW + '/Rayleigh_%.2f_%.2f' % (pick['lower_frequency'],pick['upper_frequency'])
                    if not os.path.exists(dir_phases):
                        os.makedirs(dir_phases)
                Z_trace = stream.traces[0].copy()
                if plot_modus == True:
                    Z_trace.plot(outfile= dir_SW + '/Z_comp.pdf')
                Z_trace.detrend(type="demean")
                if (pick['lower_frequency'] == float(0.0)) and (pick['upper_frequency'] == float(0.0)):
                    pass
                else:
                    Z_trace.filter('highpass', freq=pick['lower_frequency'], zerophase=True)
                    Z_trace.filter('lowpass', freq=pick['upper_frequency'], zerophase=True)
                Z_trace.detrend()
                Z_trace.detrend(type="demean")



                if plot_modus == True:
                    start_vline = int(((pick['time'].timestamp-pick['lower_uncertainty'] )- Z_trace.meta.starttime.timestamp) / Z_trace.stats.delta)
                    end_vline = int(((pick['time'].timestamp+pick['lower_uncertainty'])-Z_trace.meta.starttime.timestamp) / Z_trace.stats.delta)
                    plt.figure()
                    ax = plt.subplot(111)
                    plt.plot(Z_trace.data, alpha=0.5)
                    ymin, ymax = ax.get_ylim()
                    plt.plot(Z_trace.data)
                    plt.vlines([start_vline, end_vline], ymin, ymax)
                    plt.xlabel(Z_trace.meta.starttime.strftime('%Y-%m-%dT%H:%M:%S + sec'))
                    plt.tight_layout()
                    plt.savefig(dir_phases + '/sw_with_Rayleigh_windows.pdf')
                    # plt.show()
                    plt.close()
                Period = 1.0 /pick['frequency']
                Z_trace.trim(starttime=pick['time']-Period, endtime=pick['time']+Period)
                zero_trace = Trace(np.zeros(npts),
                                   header={"starttime":pick['time']-Period , 'delta': Z_trace.meta.delta,
                                           "station": Z_trace.meta.station,
                                           "network": Z_trace.meta.network, "location": Z_trace.meta.location,
                                           "channel": Z_trace.meta.channel})
                total_trace = zero_trace.__add__(Z_trace, method=0, interpolation_samples=0,
                                                 fill_value=Z_trace.data,
                                                 sanity_checks=False)
                Rayleigh_st.append(total_trace)
                if plot_modus == True:
                    plt.figure()
                    plt.plot(Z_trace.data, label='%.2f_%.2f' % (pick['lower_frequency'],pick['upper_frequency']))
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(dir_phases + '/diff_Love_freq.pdf')
                    plt.close()

            elif pick['phase_name'] == 'G1':
                if plot_modus == True:
                    dir_phases = dir_SW + '/Love_%.2f_%.2f' % (pick['lower_frequency'],pick['upper_frequency'])
                    if not os.path.exists(dir_phases):
                        os.makedirs(dir_phases)
                T_trace = stream.traces[2].copy()
                if plot_modus == True:
                    T_trace.plot(outfile= dir_SW + '/T_comp.pdf')
                T_trace.detrend(type="demean")
                if (pick['lower_frequency'] == float(0.0)) and (pick['upper_frequency'] == float(0.0)):
                    pass
                else:
                    T_trace.filter('highpass', freq=pick['lower_frequency'], zerophase=True)
                    T_trace.filter('lowpass', freq=pick['upper_frequency'], zerophase=True)
                T_trace.detrend()
                T_trace.detrend(type="demean")

                if plot_modus == True:
                    start_vline = int(((pick['time'].timestamp-pick['lower_uncertainty'] )- T_trace.meta.starttime.timestamp) / T_trace.stats.delta)
                    end_vline = int(((pick['time'].timestamp+pick['lower_uncertainty'] )- T_trace.meta.starttime.timestamp) / T_trace.stats.delta)
                    plt.figure()
                    ax = plt.subplot(111)
                    plt.plot(T_trace.data, alpha=0.5)
                    ymin, ymax = ax.get_ylim()
                    plt.plot(T_trace.data)
                    plt.vlines([start_vline, end_vline], ymin, ymax)
                    plt.xlabel(T_trace.meta.starttime.strftime('%Y-%m-%dT%H:%M:%S + sec'))
                    plt.tight_layout()
                    plt.savefig(dir_phases + '/sw_with_Love_windows.pdf')
                    # plt.show()
                    plt.close()
                Period = 1.0 /pick['frequency']
                T_trace.trim(starttime=pick['time']-Period, endtime=pick['time']+Period)
                zero_trace = Trace(np.zeros(npts),
                                   header={"starttime":pick['time']-Period , 'delta': T_trace.meta.delta,
                                           "station": T_trace.meta.station,
                                           "network": T_trace.meta.network, "location": T_trace.meta.location,
                                           "channel": T_trace.meta.channel})
                total_trace = zero_trace.__add__(T_trace, method=0, interpolation_samples=0,
                                                 fill_value=T_trace.data,
                                                 sanity_checks=False)
                Love_st.append(total_trace)
                if plot_modus == True:
                    plt.figure()
                    plt.plot(T_trace.data, label='%.2f_%.2f' % (pick['lower_frequency'],pick['upper_frequency']))
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(dir_phases + '/diff_Love_freq.pdf')
                    plt.close()
        return Rayleigh_st,Love_st

