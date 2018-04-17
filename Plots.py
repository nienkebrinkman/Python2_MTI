import matplotlib.pyplot as plt
import pylab
import numpy as np
import os
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
        beachball(moment, size=200, linewidth=2, facecolor='b')

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

    def make_PDF(self, sampler):
        self.sampler = sampler
        if os.path.isfile(self.sampler['filepath']) == True:
            with open(self.sampler['filepath'], 'r') as stream:
                data = yaml.load(stream)
                stream.close()
            for i in itertools.combinations(data,2):
                self.marginal_2D(data[i[0]],i[0],data[i[1]],i[1],amount_bins=20)
        else:
            print("The file does not exist yet [FIRST RUN THE MH_ALOGRITHM!]")


    def marginal_2D(self,data_x,name_x,data_y,name_y,amount_bins):
        plt.hist2d(data_x, data_y, bins=amount_bins, normed=True, cmap='binary')
        # plt.axis('equal')
        # plt.xlim([-20,40])
        # plt.ylim([10,70])
        plt.xlabel('%s' % name_x)
        plt.ylabel('%s' % name_y)
        plt.title('2D posterior marginal', fontsize = 25)
        cb = plt.colorbar()
        cb.set_label('Probability')
        plt.savefig('%s/marginal_2D_%s_%s.pdf'% (self.sampler['directory'],name_x,name_y))
        # plt.show()
        plt.close()

    def marginal_1D(self, data, name, amount_bins):

        q=np.histogram(data, bins=amount_bins)
        plt.hist(data, bins=amount_bins)
        plt.xlabel('%s' % name)
        plt.ylabel('Frequency')
        plt.show()

