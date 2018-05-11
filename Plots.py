import matplotlib.pyplot as plt
import pylab
import numpy as np
import os
import seaborn as sns
import obspy
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

        data_dict = {'Strike': data[0],
                     'Dip': data[1],
                     'Rake': data[2]}
        sns.set_style("darkgrid")
        plt.scatter(data_dict['Strike'][burnin], data_dict['Rake'][burnin], marker='^',
                    label="Start_point")
        plt.plot(data_dict['Strike'][burnin:], data_dict['Rake'][burnin:], linestyle=':',
                 label="sample_lag")
        plt.scatter(data_dict['Strike'][burnin + 1:], data_dict['Rake'][burnin + 1:], label="Sample")
        plt.xlabel("Strike")
        plt.ylabel("Rake")
        plt.legend()
        plt.savefig(dir_path)


        # data = np.transpose(np.loadtxt(filepath, delimiter=','))
        # data_dict = {'Epicentral_distance': data[0],
        #              'Depth': data[1],
        #              'Time': data[2]}

        # sns.set_style("darkgrid")
        # plt.scatter(data_dict['Epicentral_distance'][burnin], data_dict['Depth'][burnin], marker='^',
        #             label="Start_point")
        # plt.plot(data_dict['Epicentral_distance'][burnin:], data_dict['Depth'][burnin:], linestyle=':',
        #          label="sample_lag")
        # plt.scatter(data_dict['Epicentral_distance'][burnin + 1:], data_dict['Depth'][burnin + 1:], label="Sample")
        # plt.xlabel("Epicentral Distance")
        # plt.ylabel("Depth")
        # plt.legend()
        # plt.savefig(dir_path)

    def plot_seismogram_during_MH(self,ax1,ax2,ax3,d_syn,trace_window=None,savepath =None,window = False,final_plot=False):
        params = {'legend.fontsize': 'x-large',
                  'figure.figsize': (20, 15),
                  'axes.labelsize': 25,
                  'axes.titlesize': 'x-large',
                  'xtick.labelsize': 25,
                  'ytick.labelsize': 25}
        pylab.rcParams.update(params)
        if final_plot == True:
            self.traces[0].data[self.traces[0].data == 0] = np.nan
            self.traces[1].data[self.traces[1].data == 0] = np.nan
            self.traces[2].data[self.traces[2].data == 0] = np.nan

            ax1.plot(self.traces[0], linestyle=':', label="Observed data")
            ax2.plot(self.traces[1], linestyle=':')
            ax3.plot(self.traces[2], linestyle=':')
            ax1.legend()
            # plt.plot(self.d_obs, ":")
            plt.xlabel('Time [s]')
            plt.savefig(savepath.strip('.txt') + '_%i.pdf' % (self.sampler['sample_number']))
            plt.close()
        else:
            if window == True:

                trace_z = np.zeros(len(self.traces[0]))
                trace_r = np.zeros(len(self.traces[1]))
                trace_t = np.zeros(len(self.traces[2]))
                d_syn.shape = (len(d_syn))
                trace_z[trace_window['0']['P_min']:trace_window['0']['P_max']] = d_syn[0:trace_window['0']['P_len']]
                trace_z[trace_window['0']['S_min']:trace_window['0']['S_max']] = d_syn[trace_window['0']['P_len']:
                trace_window['0']['P_len'] + trace_window['0']['S_len']]
                trace_r[trace_window['1']['P_min']:trace_window['1']['P_max']] = d_syn[trace_window['0']['P_len'] +
                                                                                           trace_window['0']['S_len']:
                trace_window['0']['P_len'] + trace_window['0']['S_len'] + trace_window['1']['P_len']]
                trace_r[trace_window['1']['S_min']:trace_window['1']['S_max']] = d_syn[trace_window['0']['P_len'] +
                                                                                           trace_window['0']['S_len'] +
                                                                                           trace_window['1']['P_len']:
                trace_window['0']['P_len'] + trace_window['0']['S_len'] + trace_window['1']['P_len'] +
                trace_window['1']['S_len']]
                trace_t[trace_window['2']['P_min']:trace_window['2']['P_max']] = d_syn[trace_window['0']['P_len'] +
                                                                                           trace_window['0']['S_len'] +
                                                                                           trace_window['1']['P_len'] +
                                                                                           trace_window['1']['S_len']:
                trace_window['0']['P_len'] + trace_window['0']['S_len'] + trace_window['1']['P_len'] +
                trace_window['1']['S_len'] + trace_window['2']['P_len']]
                trace_t[trace_window['2']['S_min']:trace_window['2']['S_max']] = d_syn[trace_window['0']['P_len'] +
                                                                                           trace_window['0']['S_len'] +
                                                                                           trace_window['1']['P_len'] +
                                                                                           trace_window['1']['S_len'] +
                                                                                           trace_window['2']['P_len']:]
                trace_z[trace_z == 0] = np.nan
                trace_r[trace_r == 0] = np.nan
                trace_t[trace_t == 0] = np.nan
            else:
                trace_z = d_syn[0:len(self.traces[0])]
                trace_r = d_syn[len(self.traces[0]):len(self.traces[0]) * 2]
                trace_t = d_syn[len(self.traces[0]) * 2:len(self.traces[0]) * 3]
            ax1.plot(trace_z, alpha=0.2)
            ax2.plot(trace_r, alpha=0.2)
            ax3.plot(trace_t, alpha=0.2)



