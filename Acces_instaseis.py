import instaseis
import obspy

db=instaseis.open_db('http://instaseis.ethz.ch/marssynthetics/C30VH-BFT13-1s')
receiver= instaseis.Receiver(latitude=4.5,longitude=136.0,network="7J",station="SYNT1")
source=instaseis.Source(latitude=0.0,longitude=0.0,depth_in_m=50000,m_rr=1.710000e+24,m_tt=1.810000e+22,m_pp=-1.740000e+24,m_rt=1.990000e+23,m_rp=-1.050000e+23,m_tp=-1.230000e+24,origin_time=obspy.UTCDateTime(2020,1,2,3,4,5))
seismogram=db.get_seismograms(source=source,receiver=receiver)
seismogram.filter("highpass",freq=1.0)
seismogram.plot()