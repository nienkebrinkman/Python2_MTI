# This code will run the MCMC algorithm

# IMPORTANT: in Get_Parameters --> MCMC = 'M' or MCMC = 'MH' (see Get_Parameters for further explanation)

import instaseis

## All different classes:
from Get_Parameters import Get_Paramters
from MCMC_stream import MCMC_stream
from Create_observed import Create_observed

# Initiate Parameters:
get_parameters = Get_Paramters()

PARAMETERS = get_parameters.get_unkown()
PRIOR = get_parameters.get_prior()
VALUES = get_parameters.specifications()


## DISCUSS THIS!!!!
PRIOR['az'] = PARAMETERS['az']
PRIOR['baz'] = PARAMETERS['baz']

# Initiate the databases from instaseis:
db = instaseis.open_db(PRIOR['VELOC'])
create = Create_observed(PRIOR,PARAMETERS,db)

d_obs, traces, source = create.get_seis_automatic(prior=PRIOR,noise_model=VALUES['noise'],sdr=VALUES['sdr'])
traces_obs, p_obs, s_obs = create.get_window_obspy(traces,PARAMETERS['epi'],PARAMETERS['depth_s'],PARAMETERS['origin_time'],VALUES['npts'])
time_at_receiver = create.get_receiver_time(PARAMETERS['epi'],PARAMETERS['depth_s'],traces)
time_between_windows = create.time_between_windows(PARAMETERS['epi'],PARAMETERS['depth_s'],traces[0].meta.delta)
# traces_obs.plot()

#----------------------------------------------------------------------------------------------------------------------#

# Now we can Run a Monte Carlo algorthm:
M_algorithm = MCMC_stream(traces_obs,p_obs,s_obs, PRIOR,db,VALUES,time_at_receiver)
M_algorithm.start_MCMC(savepath=VALUES['directory'] + '/mcmctestttt.txt')



