import matplotlib.pyplot as plt
import pylab
import numpy as np
import os
import seaborn as sns
# import mplstereonet
import yaml
import itertools

from mpl_toolkits.basemap import Basemap
from obspy.imaging.beachball import beachball


class Plots:
    def plot_real_event(self, la_r,lo_r,la_s,lo_s):
        fig = plt.figure()
        m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80,
                    llcrnrlon=-180, urcrnrlon=180, lat_ts=20, resolution='l')
        # draw parallels and meridians.
        par = np.arange(-90., 91., 30.)
        label_par = np.full(len(par), True, dtype=bool)
        meridians = np.arange(-180., 181., 60.)
        label_meri = np.full(len(meridians), True, dtype=bool)
        m.drawparallels(np.arange(-90., 91., 30.), labels=label_par)
        m.drawmeridians(np.arange(-180., 181., 60.), labels=label_meri)

        mstatlon, mstatlat = m(lo_r, la_r)
        m.plot(mstatlon, mstatlat, 'k^', markersize=8)
        EQlon, EQlat = m(lo_s, la_s)
        m.plot(EQlon, EQlat, 'r*', markersize=10, zorder=10)
        plt.show()

    def Log_G(self, G):
        params = {'legend.fontsize': 'x-large',
                  'figure.figsize': (15, 15),
                  'axes.labelsize': 25,
                  'axes.titlesize': 'x-large',
                  'xtick.labelsize': 25,
                  'ytick.labelsize': 25}
        pylab.rcParams.update(params)
        plt.imshow(np.log(G))
        # plt.title('Matrix : G')
        plt.gca().set_aspect(1.0 / 10000.0)
        # plt.gca().set_aspect(10000.0)
        plt.show()

    def G_transpose_G(self, G):
        params = {'legend.fontsize': 'x-large',
                  'figure.figsize': (15, 15),
                  'axes.labelsize': 25,
                  'axes.titlesize': 'x-large',
                  'xtick.labelsize': 25,
                  'ytick.labelsize': 25}
        pylab.rcParams.update(params)
        plt.imshow(np.matmul(G.T, G))
        # plt.title('G.T G')
        plt.show()

    def Beachball(self, moment):
        params = {'legend.fontsize': 'x-large',
                  'figure.figsize': (15, 15),
                  'axes.labelsize': 25,
                  'axes.titlesize': 'x-large',
                  'xtick.labelsize': 25,
                  'ytick.labelsize': 25}
        pylab.rcParams.update(params)
        try:
            beachball(moment, size=200, linewidth=2, facecolor='b')
            plt.show()
        except TypeError:
            print ("TypeError")

    def Compare_seismograms(self, forward_data, instaseis_data):
        params = {'legend.fontsize': 'x-large',
                  'figure.figsize': (15, 15),
                  'axes.labelsize': 25,
                  'axes.titlesize': 'x-large',
                  'xtick.labelsize': 25,
                  'ytick.labelsize': 25}
        pylab.rcParams.update(params)
        fig = plt.figure(figsize=(20, 10))
        ax1 = fig.add_subplot(121)
        ax1.plot(forward_data)
        plt.title('Seismogram calculated with forward model')
        plt.xlabel('t')
        plt.ylabel('Forward data displacement [m]')
        plt.grid()

        ax2 = fig.add_subplot(122)
        ax2.plot(instaseis_data)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        plt.title('Seismogram calculated with Instaseis')
        plt.xlabel('t')
        plt.ylabel('Instaseis data displacement [m]')
        plt.grid()
        plt.show()

    def marginal_2D(self, data_x, name_x, data_y, name_y, amount_bins, directory, filename):
        params = {'legend.fontsize': 'x-large',
                  'figure.figsize': (15, 15),
                  'axes.labelsize': 25,
                  'axes.titlesize': 'x-large',
                  'xtick.labelsize': 25,
                  'ytick.labelsize': 25}
        pylab.rcParams.update(params)
        plt.hist2d(data_x, data_y, bins=amount_bins, normed=True, cmap='binary')
        # plt.hist2d(data_x, data_y, range=[[np.min(data_x), np.max(data_x)], [30.80, 30.85]], bins=amount_bins, normed=True,
        #            cmap='binary')
        # plt.axis('equal')
        # plt.xlim([-20,40])
        # plt.ylim([10,70])
        plt.xlabel('%s' % name_x)
        plt.ylabel('%s' % name_y)
        plt.title('2D posterior marginal', fontsize=25)
        cb = plt.colorbar()
        cb.set_label('Probability')
        dir_proc = directory + '/2D_margi_Plots/%s' % filename
        if not os.path.exists(dir_proc):
            os.makedirs(dir_proc)
        filepath_proc = dir_proc + '/2D_%s_%s.pdf' % (name_x, name_y)
        plt.savefig(filepath_proc)
        # plt.show()
        plt.close()

    def marginal_1D(self, data, name, amount_bins, directory, filename):
        params = {'legend.fontsize': 'x-large',
                  'figure.figsize': (15, 15),
                  'axes.labelsize': 25,
                  'axes.titlesize': 'x-large',
                  'xtick.labelsize': 25,
                  'ytick.labelsize': 25}
        pylab.rcParams.update(params)
        q = np.histogram(data, bins=amount_bins)
        plt.hist(data, bins=amount_bins)
        plt.xlabel('%s' % name)
        plt.ylabel('Frequency')
        dir_proc = directory + '/1D_margi_Plots/%s' % filename
        if not os.path.exists(dir_proc):
            os.makedirs(dir_proc)
        filepath_proc = dir_proc + '/1D_%s.pdf' % (name)
        plt.savefig(filepath_proc)
        # plt.show()
        plt.close()

    def Kernel_density(self, data, data_x, data_y, parameters, directory, savename):
        dir = directory + '/Kernel_density'
        if not os.path.exists(dir):
            os.makedirs(dir)
        # dir_path = dir + '/%s.pdf' %savename
        dir_path = dir + '/%s.pdf' % savename
        # dir_path = dir + '/Real_%s_%.2f_Real_%s_%.2f.pdf' % (data_x,parameters['%s'%data_x],data_y,parameters['%s'%data_y])
        sns.jointplot(x=data_x, y=data_y, data=data, kind="kde")
        plt.savefig(dir_path)
        plt.close()

    def Pair_Grid(self, data, directory, savename):
        dir = directory + '/Pair_grid'
        if not os.path.exists(dir):
            os.makedirs(dir)
        dir_path = dir + '/%s.pdf' % savename
        g = sns.PairGrid(data=data)
        g.map_diag(sns.kdeplot)
        g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6)
        plt.savefig(dir_path)
        plt.close()

    def sampler(self, filepath, directory, savename):
        params = {'legend.fontsize': 'x-large',
                  'figure.figsize': (15, 15),
                  'axes.labelsize': 25,
                  'axes.titlesize': 'x-large',
                  'xtick.labelsize': 25,
                  'ytick.labelsize': 25}
        pylab.rcParams.update(params)

        dir = directory + '/sampler'
        if not os.path.exists(dir):
            os.makedirs(dir)
        dir_path = dir + '/%s.pdf' % savename

        burnin = 980
        data = np.transpose(np.loadtxt(filepath, delimiter=','))
        data_dict = {'Epicentral_distance': data[0],
                     'Depth': data[1],
                     'Time': data[2]}

        sns.set_style("darkgrid")
        plt.scatter(data_dict['Epicentral_distance'][burnin], data_dict['Depth'][burnin], marker='^',
                    label="Start_point")
        plt.plot(data_dict['Epicentral_distance'][burnin:], data_dict['Depth'][burnin:], linestyle=':',
                 label="sample_lag")
        plt.scatter(data_dict['Epicentral_distance'][burnin + 1:], data_dict['Depth'][burnin + 1:], label="Sample")
        plt.xlabel("Epicentral Distance")
        plt.ylabel("Depth")
        plt.legend()
        plt.savefig(dir_path)

    def polar_plot(self):
        f, axarr = plt.subplots(2, 2, subplot_kw=dict(projection='polar'))
        axarr[0, 0].plot(x, y)
        axarr[0, 0].set_title('Axis [0,0]')
        axarr[0, 1].scatter(x, y)
        axarr[0, 1].set_title('Axis [0,1]')
        axarr[1, 0].plot(x, y ** 2)
        axarr[1, 0].set_title('Axis [1,0]')
        axarr[1, 1].scatter(x, y ** 2)
        axarr[1, 1].set_title('Axis [1,1]')
        # Fine-tune figure; make subplots farther from each other.
        f.subplots_adjust(hspace=0.3)
