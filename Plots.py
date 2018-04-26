import matplotlib.pyplot as plt
import pylab
import numpy as np
import os
import seaborn as sns
import yaml
import itertools


from obspy.imaging.beachball import beachball

class Plots:
    def __init__(self):
        params = {'legend.fontsize': 'x-large',
                  'figure.figsize': (15, 15),
                  'axes.labelsize': 25,
                  'axes.titlesize': 'x-large',
                  'xtick.labelsize': 25,
                  'ytick.labelsize': 25}
        pylab.rcParams.update(params)

    def Log_G(self, G):
        plt.imshow(np.log(G))
        # plt.title('Matrix : G')
        plt.gca().set_aspect(1.0 / 10000.0)
        # plt.gca().set_aspect(10000.0)
        plt.show()

    def G_transpose_G(self, G):
        plt.imshow(np.matmul(G.T, G))
        # plt.title('G.T G')
        plt.show()

    def Beachball(self, moment):
        try:
            beachball(moment, size=200, linewidth=2, facecolor='b')
            plt.show()
        except TypeError:
            print ("TypeError")

    def Compare_seismograms(self, forward_data, instaseis_data):
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
        dir_proc = directory +'/2D_margi_Plots/%s' % filename
        if not os.path.exists(dir_proc):
            os.makedirs(dir_proc)
        filepath_proc= dir_proc + '/2D_%s_%s.pdf' % (name_x,name_y)
        plt.savefig(filepath_proc)
        # plt.show()
        plt.close()


    def marginal_1D(self, data, name, amount_bins,directory,filename):
        q = np.histogram(data, bins=amount_bins)
        plt.hist(data, bins=amount_bins)
        plt.xlabel('%s' % name)
        plt.ylabel('Frequency')
        dir_proc = directory +'/1D_margi_Plots/%s' % filename
        if not os.path.exists(dir_proc):
            os.makedirs(dir_proc)
        filepath_proc= dir_proc + '/1D_%s.pdf' % (name)
        plt.savefig(filepath_proc)
        # plt.show()
        plt.close()

    def Kernel_density(self,data,data_x,data_y,directory,savename):
        dir= directory +'/Kernel_density/%s' % savename
        if not os.path.exists(dir):
            os.makedirs(dir)
        sns.jointplot(x=data_x, y=data_y, data=data, kind="kde")
        plt.savefig(dir)
        plt.close()

    def Pair_Grid(self,data,directory,savename):
        dir= directory +'/Pair_grid/%s' % savename
        if not os.path.exists(dir):
            os.makedirs(dir)
        g = sns.PairGrid(data=data)
        g.map_diag(sns.kdeplot)
        g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6)
        plt.savefig(dir)
        plt.close()





