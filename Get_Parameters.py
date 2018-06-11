# ---------------------------------------------------------------------------------------------------------------------#
#                                        Parameters [edit manually]                                                    #
# ---------------------------------------------------------------------------------------------------------------------#
import obspy
import numpy as np
from obspy.geodetics import kilometer2degrees
from obspy.geodetics.base import gps2dist_azimuth

class Get_Paramters:
    def specifications(self):
        ## Returns: Dict {}

        # Misfit options:
        #   - 'L2' - L2_Norm
        #   - 'CC' - Cross_correlation
        misfit = 'CC'

        # MCMC algorithm options:
        #   - 'MH' - Metropolis Hasting
        #   - 'M'  - Metropolis
        MCMC = 'M'

        # Construct the observed data:
        sdr = True # If True: invert Strike/Dip/Rake, If False: invert m_tt,m_pp,m_rr,m_tp,m_rt,m_rp
        noise = True
        plot_modus = False # If True: you make seismogram plots during your MCMC algorithm
        temperature = 1
        directory = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/Results/Tuning'
        npts = 1000 # The amount of samples of your seismogram


        SPEC={
            'sdr':sdr,
            'noise':noise,
            'plot_modus':plot_modus,
            'misfit':misfit,
            'MCMC':MCMC,
            'temperature':temperature,
            'directory':directory,
            'npts': npts}
        return SPEC


    def get_prior(self, estimated_epi=30.8309058515):
        ## Returns: Dict {}

        # PREDICTED VALUES:
        estimated_epi = estimated_epi  # [Degrees]

        # SAMPLER contains all the prior information to optimize the inversion
        PRIOR = {
            'radius':'radius',
            'la_r': 'la_r',
            'lo_r': 'lo_r',
            'M0': 'Mo',
            'network': 'network',
            'station': 'station',
            'filter': 'filter',
            'freq_filter': 'freq_filter',
            'az ': 'az',
            'baz': 'baz',
            'kind': 'displacement',
            'kernelwidth': 'kernelwidth',
            'definition': 'definition',
            'components': 'components',
            'alpha': 'alpha',
            'beta': 'beta',
            'm_ref': 'm_ref',
            'VELOC_taup': 'VELOC_taup',
            'VELOC': 'VELOC',
            'noise_model': 'noise_model',
            'strike': {'range_min': 'min', 'range_max': 'max', 'spread': 'spread'},
            'dip': {'range_min': 'min', 'range_max': 'max', 'spread': 'spread'},
            'rake': {'range_min': 'min', 'range_max': 'max', 'spread': 'spread'},
            'depth': {'range_min': 'min', 'range_max': 'max', 'step': 'spread'},
            'epi': {'range_min': 'min', 'range_max': 'max', 'spread': 'spread'},
            'sample_number': 'sample_number',
            'var_est': 'var_est'}

        # - Radius of the body used:
        # PRIOR['radius'] = 3389.5 # Mars
        PRIOR['radius'] = 6371 # Earth

        # -Receiver
        PRIOR['la_r'] = 40  # Latitude -90 <> 90
        PRIOR['lo_r'] = 20  # Longitude -180 <> 180
        PRIOR['network'] = "7J"  # Network
        PRIOR['station'] = "SYNT1"  # Station

        # -Source
        PRIOR['M0'] = 1E15
        PRIOR['components'] = ["Z", "R", "T"]

        # -filter
        PRIOR['filter'] = 'highpass'
        PRIOR['freq_filter'] = 1.0

        PRIOR['kind'] = 'displacement'
        PRIOR['kernelwidth'] = 12
        PRIOR['definition'] = 'seiscomp'

        # -Inversion parameters
        PRIOR['alpha'] = 10 ** (-24)
        PRIOR['beta'] = 10 ** (-23)
        PRIOR['m_ref'] = np.array([1.0000e+16, 1.0000e+16, 1.0000e+16, 1.0000e+16, 1.0000e+16])

        # Parameters for velocity model
        # PRIOR['VELOC'] = 'syngine://iasp91_2s'
        PRIOR['VELOC'] = "/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/Database/10s_PREM"
        PRIOR['VELOC_taup'] = 'iasp91'

        # Model used to add noise in seismogram:
        PRIOR['noise_model'] ='Tcompact' #'STS2' #

        # Sample / spread ranges for the different parameters
        PRIOR['strike']['range_min'] = 0
        PRIOR['strike']['range_max'] = 360
        PRIOR['strike']['spread'] = 1
        PRIOR['dip']['range_min'] = 0
        PRIOR['dip']['range_max'] = 90
        PRIOR['dip']['spread'] = 1
        PRIOR['rake']['range_min'] = -180
        PRIOR['rake']['range_max'] = 180
        PRIOR['rake']['spread'] = 1

        PRIOR['depth']['range_min'] = 0
        PRIOR['depth']['range_max'] = 50000
        PRIOR['depth']['spread'] = 100
        PRIOR['epi']['range_min'] = estimated_epi - 10
        PRIOR['epi']['range_max'] = estimated_epi + 10
        PRIOR['epi']['spread'] = 1

        PRIOR['sample_number'] = 100000
        PRIOR['var_est'] = 0.05 # Variance estimate 5% off the observed data

        return PRIOR

    def get_unkown(self):
        ## returns Dict {}

        # PARAMETERS describe the unkown parameters (The ones we are going to invert)
        # !only use these to create your observed data!
        PARAMETERS = {
            'la_s': 'la_s',
            'lo_s': 'lo_s',
            'depth_s': 'depth_s',
            'strike': 'strike',
            'dip': 'dip',
            'rake': 'rake',
            'm_rr': 'm_rr',
            'm_tt': 'm_tt',
            'm_pp': 'm_pp',
            'm_rt': 'm_rt',
            'm_rp': 'm_rp',
            'm_tp': 'm_tp',
            'epi': 'epi',
            'origin_time': 'origin_time',}

        # Source parameters
        PARAMETERS['la_s'] = 10
        PARAMETERS['lo_s'] = 12

        PARAMETERS['depth_s'] = 10000  # [m]
        PARAMETERS['strike'] = 79
        PARAMETERS['dip'] = 50
        PARAMETERS['rake'] = 20

        PARAMETERS['m_tt'] = 1.81e+22  # 3.81e+15
        PARAMETERS['m_pp'] = -1.74e+24  # -4.74e+17
        PARAMETERS['m_rr'] = 1.71e+24  # 4.71e+17
        PARAMETERS['m_tp'] = -1.230000e+24
        PARAMETERS['m_rt'] = 1.99e+23  # 3.99e+16
        PARAMETERS['m_rp'] = -1.05e+23  # -8.05e+16

        PARAMETERS['origin_time'] = obspy.UTCDateTime(2020, 1, 2, 3, 4, 5)

        PRIOR = self.get_prior()


        # -Greens function
        dist, az, baz = gps2dist_azimuth(lat1=PARAMETERS['la_s'],
                                         lon1=PARAMETERS['lo_s'],
                                         lat2=PRIOR['la_r'],
                                         lon2=PRIOR['lo_r'], a=PRIOR['radius'], f=0)
        PARAMETERS['baz'] = baz
        PARAMETERS['az'] = az
        PARAMETERS['epi'] = kilometer2degrees(dist, radius=PRIOR['radius'])
        return PARAMETERS
