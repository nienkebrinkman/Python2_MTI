# ---------------------------------------------------------------------------------------------------------------------#
#                                        Parameters [edit manually]                                                    #
# ---------------------------------------------------------------------------------------------------------------------#
import obspy
import numpy as np
import os
from obspy.geodetics import kilometer2degrees
from obspy.geodetics.base import gps2dist_azimuth


class Get_Paramters:
    def get_initial_par(self):
        ## Returns: Dict {}

        PARAMETERS = {
            'la_s': 'la_s',
            'la_r': 'la_r',
            'lo_s': 'lo_s',
            'network': 'network',
            'strike': 'strike',
            'dip': 'dip',
            'rake': 'rake',
            'm_rr': 'm_rr',
            'm_tt': 'm_tt',
            'm_pp': 'm_pp',
            'm_rt': 'm_rt',
            'm_rp': 'm_rp',
            'm_tp': 'm_tp',
            'origin_time': 'origin_time',
            'filter': 'filter',
            'freq_filter': 'freq_filter',
            'az ': 'az',
            'epi': 'epi',
            'kind': 'displacement',
            'kernelwidth': 'kernelwidth',
            'definition': 'definition',
            'components': 'components',
            'alpha': 'alpha',
            'beta': 'beta',
            'm_ref': 'm_ref',
            'VELOC_taup': 'VELOC_taup',
            'VELOC': 'VELOC',
            'noise_model': 'noise_model'}

        # -Receiver
        PARAMETERS['la_r'] = 40  # Latitude
        PARAMETERS['lo_r'] = 20  # Longitude
        PARAMETERS['network'] = "7J"  # Network
        PARAMETERS['station'] = "SYNT1"  # Station

        # -Source
        PARAMETERS['la_s'] = 10
        PARAMETERS['lo_s'] = 12
        PARAMETERS['depth_s'] = 10000  # [m]
        PARAMETERS['strike'] = 79
        PARAMETERS['dip'] = 50
        PARAMETERS['rake'] = 20
        PARAMETERS['M0'] = 1E17
        PARAMETERS['m_tt'] = 1.81e+22  # 3.81e+15
        PARAMETERS['m_pp'] = -1.74e+24  # -4.74e+17
        PARAMETERS['m_rr'] = 1.71e+24  # 4.71e+17
        PARAMETERS['m_tp'] = -1.230000e+24
        PARAMETERS['m_rt'] = 1.99e+23  # 3.99e+16
        PARAMETERS['m_rp'] = -1.05e+23  # -8.05e+16
        PARAMETERS['origin_time'] = obspy.UTCDateTime(2020, 1, 2, 3, 4, 5)
        PARAMETERS['components'] = ["Z", "R", "T"]

        # -filter
        PARAMETERS['filter'] = 'highpass'
        PARAMETERS['freq_filter'] = 1.0

        # -Greens function
        dist, az, baz = gps2dist_azimuth(lat1=PARAMETERS['la_s'],
                                         lon1=PARAMETERS['lo_s'],
                                         lat2=PARAMETERS['la_r'],
                                         lon2=PARAMETERS['lo_r'], a=3389.5, f=0)
        PARAMETERS['baz']=baz
        PARAMETERS['az'] = az
        PARAMETERS['epi'] = kilometer2degrees(dist, radius=3389.5)

        # PARAMETERS['epi'] = np.around(kilometer2degrees(dist, radius=3389.5),decimals=1)
        PARAMETERS['kind'] = 'displacement'
        PARAMETERS['kernelwidth'] = 12
        PARAMETERS['definition'] = 'seiscomp'

        # -Inversion parameters
        PARAMETERS['alpha'] = 10 ** (-24)
        PARAMETERS['beta'] = 10 ** (-23)
        PARAMETERS['m_ref'] = np.array([1.0000e+16, 1.0000e+16, 1.0000e+16, 1.0000e+16, 1.0000e+16])

        # Parameters for velocity model
        PARAMETERS['VELOC'] = 'http://instaseis.ethz.ch/marssynthetics/C30VH-BFT13-1s'
        PARAMETERS['VELOC_taup'] = 'iasp91'

        # Model used to add noise in seismogram:
        PARAMETERS['noise_model'] = 'STS2'

        return PARAMETERS

    def get_MHMC_par(self, directory='/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/testdata',
                     filename='trial.txt'):

        ## returns Dict {}

        filepath = os.path.join(directory, filename)

        sampler = {
            'strike': {'range_min': 'min', 'range_max': 'max', 'step': 'step'},
            'dip': {'range_min': 'min', 'range_max': 'max', 'step': 'step'},
            'rake': {'range_min': 'min', 'range_max': 'max', 'step': 'step'},
            'depth': {'range_min': 'min', 'range_max': 'max', 'step': 'step'},
            'epi': {'range_min': 'min', 'range_max': 'max', 'step': 'step'},
            'time_range': 'time_range',
            'moment_range': 'moment_range',
            'sample_number': 'sample_number',
            'var_est': 'var_est',
            'directory': directory,
            'filename': filename,
            'filepath': filepath}
        # Sample ranges if sdr = True
        sampler['strike']['range_min'] = 0
        sampler['strike']['range_max'] = 360
        sampler['strike']['step'] = 20
        sampler['dip']['range_min'] = 0
        sampler['dip']['range_max'] = 90
        sampler['dip']['step'] = 20
        sampler['rake']['range_min'] = -180
        sampler['rake']['range_max'] = 180
        sampler['rake']['step'] = 20

        # TODO - depth and epi need to be independent from parameters!!!
        PARAMETERS = self.get_initial_par()
        sampler['depth']['range_min'] = PARAMETERS['depth_s'] - 5000
        sampler['depth']['range_max'] = PARAMETERS['depth_s'] + 5000
        sampler['depth']['step'] = 20
        sampler['epi']['range_min'] = PARAMETERS['epi'] - 10
        sampler['epi']['range_max'] = PARAMETERS['epi'] + 10
        sampler['epi']['step'] = 40
        sampler['time_range'] = PARAMETERS['origin_time'].second + 30  # The time range can only vary may be two hours!!
        sampler['sample_number'] = 100
        sampler['var_est'] = 0.05
        return sampler
