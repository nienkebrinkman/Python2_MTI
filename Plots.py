import matplotlib.pyplot as plt
import pylab
import numpy as np
import os
import seaborn as sns
import obspy
from matplotlib.dates import date2num
# import mplstereonet
import yaml
import itertools

from mpl_toolkits.basemap import Basemap
from obspy.imaging.beachball import beachball


class Plots:
    def plot_real_event(self, la_r,lo_r,la_s,lo_s):
        mars_dir = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/mars_pictures/Mars_lightgray.jpg'

        fig = plt.figure()

        m= Basemap(projection='moll',lon_0=round(0.0))

        # draw parallels and meridians.
        par = np.arange(-90, 90, 30)
        label_par = np.full(len(par), True, dtype=bool)
        meridians = np.arange(-180, 180, 30)
        label_meri = np.full(len(meridians), True, dtype=bool)

        m.drawmeridians(np.arange(-180, 180, 30),labels=label_meri)
        m.drawparallels(np.arange(-90, 90, 30),label=label_par)


        m.warpimage(mars_dir)
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

    def marginal_2D(self, data_x, name_x, data_y, name_y, amount_bins, directory, filename,true_time):
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
        if name_x == 'Time':
            or_time=obspy.UTCDateTime(true_time[0], true_time[1], true_time[2], true_time[3], true_time[4], true_time[5])
            plt.xlabel(or_time.strftime('%Y-%m-%dT%H:%M:%S + sec'))
            plt.ylabel('%s' % name_y)
        elif name_y == 'Time':

            or_time=obspy.UTCDateTime(true_time[0], true_time[1], true_time[2], true_time[3], true_time[4], true_time[5])
            plt.ylabel(or_time.strftime('%Y-%m-%dT%H:%M:%S + sec'))
            plt.xlabel('%s' % name_x)
        else:
            plt.ylabel('%s' % name_y)
            plt.xlabel('%s' % name_x)


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

    def marginal_1D(self, data, name, amount_bins, directory, filename, true_time):
        params = {'legend.fontsize': 'x-large',
                  'figure.figsize': (15, 15),
                  'axes.labelsize': 25,
                  'axes.titlesize': 'x-large',
                  'xtick.labelsize': 25,
                  'ytick.labelsize': 25}
        pylab.rcParams.update(params)
        q = np.histogram(data, bins=amount_bins)
        plt.hist(data, bins=amount_bins)
        if name == 'Time':
            or_time=obspy.UTCDateTime(true_time[0], true_time[1], true_time[2], true_time[3], true_time[4], true_time[5])
            plt.xlabel(or_time.strftime('%Y-%m-%dT%H:%M:%S + sec'))
        else:
            plt.xlabel('%s' % name)
        plt.ylabel('Frequency')
        if name == 'Time':
            labels = np.ones_like(data)
            for i,v in enumerate(data):
                labels[i] = obspy.UTCDateTime(data[i])
            plt.xticks(data, labels, rotation='vertical')
        dir_proc = directory + '/1D_margi_Plots/%s' % filename
        if not os.path.exists(dir_proc):
            os.makedirs(dir_proc)
        filepath_proc = dir_proc + '/1D_%s.pdf' % (name)
        plt.savefig(filepath_proc)
        # plt.show()
        plt.close()

    def Kernel_density(self, data, data_x, data_y, directory, savename):
        dir = directory + '/Kernel_density'
        if not os.path.exists(dir):
            os.makedirs(dir)
        # dir_path = dir + '/%s.pdf' %savename
        dir_path = dir + '/%s_%s_%s.pdf' % (savename, data_x,data_y)
        h= sns.jointplot(x=data_x, y=data_y, data=data, kind="kde")
        plt.tight_layout()
        plt.savefig(dir_path)
        plt.close()

    def hist(self, data, data_x, data_y, directory, savename):
        dir = directory + '/Historgrams'
        if not os.path.exists(dir):
            os.makedirs(dir)
        dir_path = dir + '/%s_%s_%s.pdf' % (savename, data_x,data_y)
        h= sns.jointplot(x=data_x, y=data_y, data=data, kind="hex")
        plt.tight_layout()
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
        plt.tight_layout()
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


        burnin = 980
        data = np.transpose(np.loadtxt(filepath, delimiter=','))

        data_dict = {'epicentral_distance':data[0],
                     'depth':data[1],
                     'Strike': data[4],
                     'Dip': data[5],
                     'Rake': data[6]}
        sns.set_style("darkgrid")
        for i in itertools.combinations(data_dict, 2):
            plt.scatter(data_dict[i[0]][burnin], data_dict[i[1]][burnin], marker='^',
                        label="Start_point")
            plt.plot(data_dict[i[0]][burnin:], data_dict[i[1]][burnin:], linestyle=':',
                     label="sample_lag")
            plt.scatter(data_dict[i[0]][burnin + 1:], data_dict[i[1]][burnin + 1:], label="Sample")
            plt.xlabel(i[0])
            plt.ylabel(i[1])
            plt.legend()
            dir_path = dir + '/%s_%s_%s.pdf' % (savename,i[0],i[1])
            plt.savefig(dir_path)
            plt.close()

    def plot_seismogram_during_MH(self,ax,data,savepath,final_plot=False):
        params = {'legend.fontsize': 'x-large',
                  'figure.figsize': (20, 15),
                  'axes.labelsize': 25,
                  'axes.titlesize': 'x-large',
                  'xtick.labelsize': 25,
                  'ytick.labelsize': 25}
        pylab.rcParams.update(params)
        if final_plot == True:
            ax.plot(data, linestyle=':', label="Observed data")
            ax.legend()
            plt.tight_layout()
            # plt.plot(self.d_obs, ":")
            plt.xlabel('Time [s]')
            # plt.show()
            plt.savefig(savepath.replace('.txt','.pdf') )
            plt.close()

        else:
            ax.plot(data, alpha=0.2)






