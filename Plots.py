import matplotlib.pyplot as plt
import pylab
import numpy as np
from obspy.imaging.beachball import beachball

class Plots:
    def __init__(self):
        params = {'legend.fontsize': 'x-large',
                  'figure.figsize': (10, 10),
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
