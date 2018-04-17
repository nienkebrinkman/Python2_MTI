# Test script for using package PyMC:

import numpy as np
import os.path
import yaml
import pymc3 as pm
import matplotlib.pyplot as plt
import theano.tensor as tt

from Inversion_problems import Inversion_problem
from Misfit import Misfit
from scipy import optimize



directory = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/testdata'
filename = 'pymc_trial_1.yaml'
filepath = os.path.join(directory, filename)


SAMPLER = {'a': {'value': 'value', 'rane_min': 'range_min', 'range_max': 'range_max'},
              'b': {'value': 'value', 'rane_min': 'range_min', 'range_max': 'range_max'},
              'c': {'value': 'value', 'rane_min': 'range_min', 'range_max': 'range_max'},
              'sampler':'sampler',
              'amount_param':'amount_param',
              'X_range':'X_range',
              'directory':directory,
              'file_name':filename,
              'filepath':filepath}
SAMPLER['a']['value']=1
SAMPLER['a']['range_min']=-60
SAMPLER['a']['range_max']=60
SAMPLER['b']['value']=0
SAMPLER['b']['range_min']=-60
SAMPLER['b']['range_max']=60
SAMPLER['c']['value']=2
SAMPLER['c']['range_min']=-60
SAMPLER['c']['range_max']=60
SAMPLER['sampler']=1000
SAMPLER['amount_param']=3
SAMPLER['var_est']=0.05
SAMPLER['X_range'] = np.linspace(-5,5,11)
d_obs = SAMPLER['a']['value'] * SAMPLER['X_range'] ** 2 + SAMPLER['b']['value'] * SAMPLER['X_range'] + SAMPLER['c']['value']

def main():

    MH = MCMC_algorithm(SAMPLER, d_obs)
    model=MH.Hamiltonian_NUTS()


class MCMC_algorithm:
    def __init__(self, sampler, d_obs):
        # Sampler should contain the information for your Metropolis Hasting:
        # Dict: ['param_1']['range_min'] : The minimum value of a range of a certain parameter (int)
        #       ['param_1']['range_max'] : The maximum value of a range of a certain parameter (int)
        #       ['param_1']['value']     : The value of a certain parameter (int,float)
        #       ['sampler']              : The amount of samples you would like to run (e.g. 100000) (int)
        #       ['amount_param']         : The amount of parameters (int)

        self.sampler = sampler
        self.d_obs = d_obs    # The observed data

    def Metropolis_Hastings(self):
        # True parameter values
        a, b, c = 1, 0, 2
        sigma = 0.01

        # Size of dataset
        size = 100

        # Predictor variable
        X1 = np.random.randn(size)

        # Simulate outcome variable
        Y_obs = a * X1 ** 2 + b * X1 + c + np.random.randn(size) * sigma
        # uni= uniform.rvs(size=size)
        fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
        axes[0].scatter(X1, Y_obs)
        axes[0].set_ylabel('Y');
        axes[0].set_xlabel('X1');
        basic_model= pm.Model()
        plt.show()

        with basic_model:
            # Priors for unknown model parameters
            a = pm.Uniform('a', lower=self.sampler['a']['range_min'], upper=self.sampler['a']['range_max'])
            b = pm.Uniform('b', lower=self.sampler['b']['range_min'], upper=self.sampler['b']['range_max'])
            c = pm.Uniform('c', lower=self.sampler['c']['range_min'], upper=self.sampler['c']['range_max'])


            sigma = pm.HalfNormal('sigma', sd=1)

            # Expected value of outcome
            mu = a * X1 ** 2 + b * X1 + c

            # Likelihood (sampling distribution) of observations
            Y_posterior = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y_obs)



            trace = pm.sample(100000, pm.Metropolis())

            # trace = pm.sample(5000)

            # # obtain starting values via MAP
            # start = pm.find_MAP(model=basic_model)
            #
            # # instantiate sampler
            # step = pm.Slice()
            #
            # # draw 5000 posterior samples
            # trace = pm.sample(5000, step=step, start=start)
        _ = pm.traceplot(trace)
        # plt.plot(trace['a'])
        plt.show()

    def Hamiltonian_NUTS(self):
        # True parameter values
        a, b, c = 1, 0, 2
        sigma = 0.01

        # Size of dataset
        size = 100

        # Predictor variable
        # X1 = np.random.randn(size)
        X1 = np.random.randn(size)

        # Simulate outcome variable
        Y_obs = a * X1 ** 2 + b * X1 + c + np.random.randn(size) * sigma
        # uni= uniform.rvs(size=size)
        fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
        axes[0].scatter(X1, Y_obs)
        axes[0].set_ylabel('Y');
        axes[0].set_xlabel('X1');
        basic_model= pm.Model()

        with basic_model:
            # Priors for unknown model parameters
            a = pm.Uniform('a', lower=self.sampler['a']['range_min'], upper=self.sampler['a']['range_max'])
            b = pm.Uniform('b', lower=self.sampler['b']['range_min'], upper=self.sampler['b']['range_max'])
            c = pm.Uniform('c', lower=self.sampler['c']['range_min'], upper=self.sampler['c']['range_max'])
            #

            sigma = pm.HalfNormal('sigma', sd=10)

            # Expected value of outcome
            mu = a * X1 ** 2 + b * X1 + c

            # Likelihood (sampling distribution) of observations
            Y_posterior = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y_obs)

            # stds = np.ones(basic_model.ndim)
            # args = {'scaling': stds ** 2, 'is_cov': True}
            # trace = pm.sample(5000, init='jitter+adapt_diag' , tune=100,nuts_kwargs=dict(target_accept=.85))
            # trace = pm.sample(5000, init='advi+adapt_diag' , tune=100,nuts_kwargs=dict(target_accept=.85))
            trace = pm.sample(5000, init='advi+adapt_diag' , tune=100)
        _ = pm.traceplot(trace)
        # plt.plot(trace['a'])
        plt.show()

    def Instaseis(self,parameters,db):
        self.par = parameters
        self.db = db
        basic_model = pm.Model()
        with basic_model:
            # Priors for unknown model parameters
            azimuth = pm.Uniform('azimuth', lower=self.sampler['azimuth']['range_min'], upper=self.sampler['azimuth']['range_max'])
            epi = pm.Uniform('epi', lower=self.sampler['epi']['range_min'], upper=self.sampler['epi']['range_max'])
            depth = pm.Uniform('depth', lower=self.sampler['depth']['range_min'], upper=self.sampler['depth']['range_max'])
            # time = pm.Uniform('time', lower=self.sampler['depth']['range_min'], upper=self.sampler['depth']['range_max'])
            time = self.par['origin_time']

            sigma = pm.HalfNormal('sigma', sd=1)

            # Expected value of outcome
            mu = self.G_function(epi.tag.test_value,depth.tag.test_value,azimuth.tag.test_value,time)

            # Likelihood (sampling distribution) of observations
            Y_posterior = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=self.d_obs)

            # trace = pm.sample(5000, init='advi+adapt_diag' , tune=100)
            trace = pm.sample(5000, pm.Metropolis())
        _ = pm.traceplot(trace)
        plt.show()

        a=1

    def generate_G(self, epi, depth, azimuth, t):
        gf = self.db.get_greens_function(epicentral_distance_in_degree=epi, source_depth_in_m=depth, origin_time=t,
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
        G[0:G_z, 0] = zss * (0.5) * np.cos(2 * np.deg2rad(azimuth)) - zdd * 0.5
        G[0:G_z, 1] = - zdd * 0.5 - zss * (0.5) * np.cos(2 * np.deg2rad(azimuth))
        # G[0:G_z, 1] =  zdd * (1/3) + zep * (1/3)
        G[0:G_z, 2] = zss * np.sin(2 * np.deg2rad(azimuth))
        G[0:G_z, 3] = -zds * np.cos(np.deg2rad(azimuth))
        G[0:G_z, 4] = -zds * np.sin(np.deg2rad(azimuth))

        G[G_z:G_r, 0] = rss * (0.5) * np.cos(2 * np.deg2rad(azimuth)) - rdd * 0.5
        G[G_z:G_r, 1] = -0.5 * rdd - rss * (0.5) * np.cos(2 * np.deg2rad(azimuth))
        # G[G_z:G_r, 1] =  rdd * (1/3) + rep * (1/3)
        G[G_z:G_r, 2] = rss * np.sin(2 * np.deg2rad(azimuth))
        G[G_z:G_r, 3] = -rds * np.cos(np.deg2rad(azimuth))
        G[G_z:G_r, 4] = -rds * np.sin(np.deg2rad(azimuth))

        G[G_r:G_t, 0] = -tss * (0.5) * np.sin(2 * np.deg2rad(azimuth))
        G[G_r:G_t, 1] = tss * (0.5) * np.sin(2 * np.deg2rad(azimuth))
        # G[G_r:G_t, 1] =   0
        G[G_r:G_t, 2] = tss * np.cos(2 * np.deg2rad(azimuth))
        G[G_r:G_t, 3] = tds * np.sin(2 * np.deg2rad(azimuth))
        G[G_r:G_t, 4] = -tds * np.cos(2 * np.deg2rad(azimuth))
        return G

    def G_function(self, epi, depth, azimuth, t):
        G = self.generate_G(epi, depth, azimuth, t)
        inv = Inversion_problem(self.d_obs, G, self.par)
        moment = inv.Solve_damping_smoothing()
        ## -- choose a range for moment with the help of the Resolution Matrix --
        ## NOT FINISHED YET!!!
        d_syn = np.matmul(G, moment)
        return d_syn



if __name__ == '__main__':
    main()