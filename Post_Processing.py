# ---------------------------------------------------------------------------------------------------------------------#
#                                                 Post Processing                                                      #
# ---------------------------------------------------------------------------------------------------------------------#
from Plots import Plots
import os
import numpy as np
import itertools
import glob
import pandas as pd
import matplotlib.pylab as plt
import mplstereonet
import yaml
from obspy.imaging.beachball import aux_plane
from pandas.plotting import autocorrelation_plot
from obspy.imaging.beachball import beachball
from pandas.plotting import scatter_matrix
import pylab
import obspy

from Get_Parameters import Get_Paramters


def main():
    ## Post - Processing [processing the results from inversion]
    result = Post_processing_sdr()

    directory = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/Euler_data'
    path_to_file = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/Euler_data/Trials/Trial_close_GOOD/close_small_spread.txt'
    # path_to_stream = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/Blindtest/bw_reject.mseed'
    # path= '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/Additional_scripts/Iteration_runs/Together'

    savename = 'Trials'
    show = False  # Choose True for direct show, choose False for saving
    skiprows = 70
    column_names= ["Epi", "Depth", "Strike", "Dip", "Rake", "Total_misfit","S_z","S_r","S_t","P_z","P_r","BW_misfit","Rtot","Ltot"]
    # column_names= ["Epi", "Depth", "Strike", "Dip", "Rake","Total_misfit","S_z","S_r","S_t"]
    # path_to_file, save = result.convert_txt_folder_to_yaml(path, savename)
    # result.get_stereonets(filepath=path_to_file, savename=savename, directory=directory, show=show)

    # result.plot_streams(stream_filepath=path_to_stream,filepath=path_to_file,savename=savename, directory=directory,skiprows=skiprows ,column_names=column_names)

    par = Get_Paramters()
    REAL = par.get_unkown()
    PRIOR = par.get_prior()
    result.event_plot(PRIOR['la_r'],PRIOR['lo_r'],REAL['la_s'],REAL['lo_s'])
    # result.get_beachballs(REAL['strike'], REAL['dip'],REAL['rake'],PRIOR['M0'],directory+'/beachball.pdf')
    # result.trace_density(filepath=path_to_file, savename=savename, directory=directory, skiprows=skiprows, column_names=column_names,show=show)
    # result.get_accepted_samples(filepath=path_to_file,savename=savename,directory=directory, column_names,skiprows=skiprows)
    # result.get_convergence(filepath=path_to_file, savename=savename, directory=directory, skiprows=skiprows, column_names=column_names,show=show)
    # result.Seaborn_plots(filepath=path_to_file, savename=savename, directory=directory, skiprows=skiprows, column_names= column_names,show=show)
    # result.combine_all(filepath=path_to_file, savename=savename, directory=directory,skiprows=skiprows, column_names=column_names, show=show)
    # result.get_pdf(filepath=path_to_file, savename=savename, directory=directory, skiprows=skiprows, column_names=column_names,show=show)


class Post_processing_sdr:
    def convert_txt_file_to_yaml(self, filepath):
        save_path = filepath.replace('.txt', '.yaml')
        if os.path.isfile(save_path) == True:
            print("File is already converted to .yaml file, located in \n %s" % save_path)
            return save_path, save_path
        with open(filepath) as infile:
            dat = np.genfromtxt(infile, delimiter=',', skip_header=34, skip_footer=1)
            # par = open(fname, "r").readlines()[:34]
            # parameters = self.write_to_dict(par)
        # data_file = {'data': dat, 'parameters': parameters}
        data_file = {'data': dat}
        with open(save_path, 'w') as yaml_file:
            yaml.dump(data_file, yaml_file, default_flow_style=False)
        yaml_file.close()
        savename = save_path.split('/')[-1].strip('.yaml')
        return save_path, savename

    def convert_txt_folder_to_yaml(self, dir_of_txt_files, savename):
        filenames = glob.glob("%s/*.txt" % dir_of_txt_files)
        save_path = dir_of_txt_files + '/%s.yaml' % savename
        if os.path.isfile(save_path) == True:
            return save_path, savename
        for i, fname in enumerate(filenames):

            with open(fname) as infile:
                if i == 0:
                    # dat = np.genfromtxt(infile, delimiter=',',skip_footer=1)
                    dat = np.genfromtxt(infile, delimiter=',', skip_header=35, skip_footer=1)
                    # par = open(fname, "r").readlines()[:34]
                    # parameters = self.write_to_dict(par)

                else:
                    dat = np.vstack((dat, np.genfromtxt(infile, delimiter=',', skip_header=35, skip_footer=1)))
                    # dat = np.vstack((dat, np.genfromtxt(infile, delimiter=',',skip_footer=1)))
            # data_file = {'data': dat, 'parameters': parameters}
            data_file = {'data': dat}
            with open(save_path, 'w') as yaml_file:
                yaml.dump(data_file, yaml_file, default_flow_style=False)
            yaml_file.close()
            return save_path, savename

    def get_accepted_samples(self,filepath,savename,directory,skiprows,column_names):
        dir = directory + '/%s' % (savename)
        if not os.path.exists(dir):
            os.makedirs(dir)

        if filepath.endswith('.yaml') == True:
            with open(filepath, 'r') as stream:
                data_file = yaml.load(stream)
                stream.close()
                data = data_file['data']
                # parameters = data_file['parameters']
        else:
            # parameters = open(filepath, "r").readlines()[:33]
            data = np.loadtxt(filepath, delimiter=',', skiprows=skiprows)
        length = len(data[0]) - (len(column_names) + 3)
        R_length = int(data[0][-2])
        L_length = int(data[0][-1])

        for i in range(R_length):
            column_names = np.append(column_names,"R_%i" % (i+1))
        for i in range(L_length):
            column_names = np.append(column_names,"L_%i" % (i+1))
        column_names = np.append(column_names,"Accepted")
        column_names = np.append(column_names,"Rayleigh_length")
        column_names = np.append(column_names,"Love_length")

        df = pd.DataFrame(data,
                          columns=column_names)
        total = len(df["Accepted"])
        accepted = np.sum(df["Accepted"]==1)
        savepath = dir + '/Accept_%i_outof_%i.txt' % (accepted,total)
        with open(savepath, 'w') as save_file:
            save_file.write("%i,%i\n\r" % (total,accepted))
        save_file.close()
        print("Total amount of samples = %i" % len(df["Accepted"]))
        print("Total amount of accepted samples = %i" % np.sum(df["Accepted"]==1))

    def get_convergence(self, filepath, savename, directory,skiprows, column_names,show=True):
        dir = directory + '/%s' % (savename)
        if not os.path.exists(dir):
            os.makedirs(dir)

        if filepath.endswith('.yaml') == True:
            with open(filepath, 'r') as stream:
                data_file = yaml.load(stream)
                stream.close()
                data = data_file['data']
                # parameters = data_file['parameters']
        else:
            # parameters = open(filepath, "r").readlines()[:33]
            data = np.loadtxt(filepath, delimiter=',', skiprows=skiprows)

        length = len(data[0]) - (len(column_names) + 3)
        R_length = int(data[0][-2])
        L_length = int(data[0][-1])

        for i in range(R_length):
            column_names = np.append(column_names,"R_%i" % (i+1))
        for i in range(L_length):
            column_names = np.append(column_names,"L_%i" % (i+1))
        column_names = np.append(column_names,"Accepted")
        column_names = np.append(column_names,"Rayleigh_length")
        column_names = np.append(column_names,"Love_length")
        #
        df = pd.DataFrame(data,
                          columns=column_names)
        # df = pd.DataFrame(data,
        #                   columns=["Epicentral_distance", "Depth", "Strike", "Dip", "Rake", "Total_misfit","BW_misfit","R_misfit","L_misfit"])
        plt.figure(1)
        ax = plt.subplot(111)
        ax.plot(np.arange(0, len(df['Total_misfit'])), df['Total_misfit'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('-Log(likelihood)')
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        plt.tight_layout()
        if show == True:
            plt.show()
        else:
            plt.savefig(dir + '/Convergence.pdf')
            plt.close()

    def combine_all(self, filepath, savename, directory, skiprows, column_names,show=True):
        dir = directory + '/%s' % (savename.strip('.yaml'))
        if not os.path.exists(dir):
            os.makedirs(dir)

        if filepath.endswith('.yaml') == True:
            with open(filepath, 'r') as stream:
                data_file = yaml.load(stream)
                stream.close()
                data = data_file['data']
                # parameters = data_file['parameters']
        else:
            # parameters = open(filepath, "r").readlines()[:33]
            data = np.loadtxt(filepath, delimiter=',', skiprows=skiprows)
        length = len(data[0]) - (len(column_names) + 3)
        L_length = int(data[0][-1])
        R_length = int(data[0][-2])

        for i in range(R_length):
            column_names = np.append(column_names,"R_%i" % (i+1))
        for i in range(L_length):
            column_names = np.append(column_names,"L_%i" % (i+1))
        column_names = np.append(column_names,"Accepted")
        column_names = np.append(column_names,"Rayleigh_length")
        column_names = np.append(column_names,"Love_length")

        par=Get_Paramters()
        REAL = par.get_unkown()

        df = pd.DataFrame(data,
                          columns=column_names)
        df_select = df[["Epi", "Depth", "Strike", "Dip", "Rake","M0"]]
        fig = plt.figure(figsize=(20,20))
        ax1 = plt.subplot2grid((5,2),(0,0),colspan=2)
        # ax1.axhline(y = REAL['epi'] ,linewidth = 0.3 , linestyle =':', color = 'b')
        ax1.plot(df_select['Epi'],label="Epi" ,c = 'b')
        ax1.tick_params("y", colors='b')
        ax1.set_ylabel('Epicentral distance [degree]', color='b')
        ax2 = ax1.twinx()
        ax2.plot(df_select['Depth'],label="Depth",c ='r')
        # ax2.axhline(y=REAL['depth_s'], linewidth=0.3, linestyle=':', color='r')
        ax2.tick_params('y', colors='r')
        ax2.set_ylabel('Depth [m]', color='r')
        plt.tight_layout()
        ax3 = plt.subplot2grid((5,2), (1,0),colspan=2)
        ax3.plot(df_select['Strike'], label = "Strike", color = "b")
        # ax3.axhline(y=REAL['strike'], linewidth=0.3, linestyle=':', color='b')
        # ax3.axhline(y=REAL['dip'], linewidth=0.3, linestyle=':', color='g')
        # ax3.axhline(y=REAL['rake'], linewidth=0.3, linestyle=':', color='r')
        ax3.plot(df_select['Dip'], label = "Dip" ,color = 'g')
        ax3.plot(df_select['Rake'], label = "Rake", color = 'r')
        ax3.set_ylabel('Moment tensor angle [degree]', color='k')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        axes = ax3.twinx()
        axes.plot(df_select['M0'],label="M0",c ='m')
        # ax2.axhline(y=REAL['depth_s'], linewidth=0.3, linestyle=':', color='r')
        axes.tick_params('y', colors='m')
        axes.set_ylabel('M0', color='r')
        plt.yscale('log')
        plt.tight_layout()

        R_select = df.filter(like='R_')
        L_select = df.filter(like='L_')
        df_select_xi = df[["Total_misfit","S_z","S_r","S_t","P_z","P_r","BW_misfit","Rtot","Ltot"]]
        ax4 = plt.subplot2grid((5, 2), (2, 0), colspan=2)
        ax4.plot(df_select_xi['Total_misfit'], label = "Total_misfit",c='r')
        # ax4.plot(df_select_xi['S_z'], label = "S_z")
        # ax4.plot(df_select_xi['S_r'], label = "S_r")
        # ax4.plot(df_select_xi['S_t'], label = "S_t")
        # ax4.plot(df_select_xi['P_z'], label = "P_z")
        # ax4.plot(df_select_xi['P_r'], label = "P_r")
        ax4.plot(df_select_xi['BW_misfit'], label = "BW_tot")
        ymin, ymax = ax4.get_ylim()
        plt.yscale('log')
        # plt.ylim((pow(10, 0), pow(10, 3)))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        ax5=plt.subplot2grid((5,2),(3,0),colspan=2)
        # for i in R_select:
        #     ax5.plot(R_select[i],label = i)
        ax5.plot(df_select_xi['Rtot'], label = "Rtot")
        plt.yscale('log')
        # plt.ylim((pow(10, 0), pow(10, 4)))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        ax6 = plt.subplot2grid((5,2),(4,0),colspan=2)
        # for i in L_select:
        #     ax6.plot(L_select[i],label = i)
        ax6.plot(df_select_xi['Ltot'], label = "Ltot")
        plt.yscale('log')

        # plt.ylim((pow(10, 0), pow(10, 4)))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(dir+'/combined_all_par.pdf')
        plt.close()

    def trace_density(self, filepath, savename, directory,skiprows, column_names, show=True):
        dir = directory + '/%s' % (savename.strip('.yaml'))
        if not os.path.exists(dir):
            os.makedirs(dir)

        if filepath.endswith('.yaml') == True:
            with open(filepath, 'r') as stream:
                data_file = yaml.load(stream)
                stream.close()
                data = data_file['data']
                # parameters = data_file['parameters']
        else:
            # parameters = open(filepath, "r").readlines()[:33]
            data = np.loadtxt(filepath, delimiter=',', skiprows=skiprows)

        length = len(data[0]) - (len(column_names) + 3)
        L_length = 4#int(data[0][-1])
        R_length = 4#int(data[0][-2])

        for i in range(R_length):
            column_names = np.append(column_names,"R_%i" % (i+1))
        for i in range(L_length):
            column_names = np.append(column_names,"L_%i" % (i+1))
        column_names = np.append(column_names,"Accepted")
        # column_names = np.append(column_names,"Rayleigh_length")
        # column_names = np.append(column_names,"Love_length")

        params = {'legend.fontsize': 'x-large',
                  'figure.figsize': (15, 15),
                  'axes.labelsize': 25,
                  'axes.titlesize': 'x-large',
                  'xtick.labelsize': 25,
                  'ytick.labelsize': 25}
        pylab.rcParams.update(params)
        #
        df = pd.DataFrame(data,
                          columns=column_names)
        df_select = df[["Epi", "Depth", "Strike", "Dip", "Rake"]]
        par=Get_Paramters()
        REAL = par.get_unkown()
        real_v=np.array([REAL['epi'],REAL['depth_s'],REAL['strike'], REAL['dip'],REAL['rake']])
        # real_v=np.array([35.2068855191,16848.1405882,99.88000245897199, 77.02994881296266,68.80551508147109])
        # real_v=np.array([73.8059395565,26247.7456326,158.92744919732135, 47.46909146946276,-45.69139707143311])
        strike,dip,rake = aux_plane(REAL['strike'],REAL['dip'],REAL['rake'])
        # strike,dip,rake = aux_plane(99.88000245897199,77.02994881296266,68.80551508147109)
        # strike,dip,rake = aux_plane(158.92744919732135, 47.46909146946276,-45.69139707143311)


        fig = plt.figure(figsize=(20,20))
        row = 0

        for i in df_select:
            ax1 = plt.subplot2grid((5,2),(row,0))
            ax1.plot(df_select[i],label=i)
            # ax1.axhline(y=REAL[df_select], linewidth=0.3, linestyle=':')
            ax1.set_title("Trace %s" %i ,color= 'b',fontsize = 20)
            ax1.set_xlabel("Iteration")
            # ax1.set_ylabel("Epicentral ")
            plt.tight_layout()
            ax2 = plt.subplot2grid((5,2), (row, 1))
            plt.hist(df_select[i],bins=100)
            ymin, ymax = ax2.get_ylim()
            plt.vlines(df_select[i][1],ymin=ymin,ymax=ymax,colors='r',linewidth = 3)
            plt.vlines(real_v[row],ymin=ymin,ymax=ymax,colors='k',linewidth = 3)
            if i == 'Strike':
                plt.vlines(strike,ymin=ymin,ymax=ymax,colors='g',linewidth = 3)
            if i == 'Dip':
                plt.vlines(dip,ymin=ymin,ymax=ymax,colors='g',linewidth = 3)
            if i == 'Rake':
                plt.vlines(rake,ymin=ymin,ymax=ymax,colors='g',linewidth = 3)

            # y, binEdges = np.histogram(df_select[i], bins=100)
            # bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
            # pylab.plot(bincenters, y, '-',label = "%s" % i)
            ax2.set_title("Density %s"%i,color= 'b',fontsize = 20)
            ax2.set_xlabel("N=%i" % (len(df_select[i])))
            plt.tight_layout()
            row += 1
        plt.savefig(dir+ '/Trace_density.pdf')

    def get_pdf(self, filepath, savename, directory, skiprows, column_names,show=True):
        dir = directory + '/%s' % (savename.strip('.yaml'))
        if not os.path.exists(dir):
            os.makedirs(dir)

        dir_pdf = dir + '/PDF'
        if not os.path.exists(dir_pdf):
            os.makedirs(dir_pdf)

        if filepath.endswith('.yaml') == True:
            with open(filepath, 'r') as stream:
                data_file = yaml.load(stream)
                stream.close()
                data = data_file['data']
                # parameters = data_file['parameters']
        else:
            # parameters = open(filepath, "r").readlines()[:33]
            data = np.loadtxt(filepath, delimiter=',', skiprows=skiprows)
        length = len(data[0]) - (len(column_names) + 3)
        R_length = int(data[0][-2])
        L_length = int(data[0][-1])

        for i in range(R_length):
            column_names = np.append(column_names,"R_%i" % (i+1))
        for i in range(L_length):
            column_names = np.append(column_names,"L_%i" % (i+1))
        column_names = np.append(column_names,"Accepted")
        column_names = np.append(column_names,"Rayleigh_length")
        column_names = np.append(column_names,"Love_length")

        df = pd.DataFrame(data,
                          columns=column_names)
        df_select = df[["Epi", "Depth", "Strike", "Dip", "Rake"]]
        pdf = Plots()
        for i, value in df_select.iteritems():
            pdf.marginal_1D(data=value, name=i, amount_bins=20, directory=dir_pdf, show=show)
        for i in itertools.combinations(df_select, 2):
            pdf.marginal_2D(df_select[i[0]], i[0], df_select[i[1]], i[1], amount_bins=20, directory=dir_pdf, show=show)

    def get_beachballs(self,strike,dip,rake,M0,savepath):
        rdip = np.deg2rad(dip)
        rstr = np.deg2rad(strike)
        rrake = np.deg2rad(rake)

        nx = -np.sin(rdip) * np.sin(rstr)
        ny = np.sin(rdip) * np.cos(rstr)
        nz = -np.cos(rdip)

        dx = np.cos(rrake) * np.cos(rstr) + np.cos(rdip) * np.sin(rrake) * np.sin(rstr)
        dy = np.cos(rrake) * np.sin(rstr) - np.cos(rdip) * np.sin(rrake) * np.cos(rstr)
        dz = -np.sin(rdip) * np.sin(rrake)

        # dx =  np.cos(rrake)*np.cos(rstr)+np.cos(rdip)*np.sin(rrake)*np.sin(rstr)
        # dy =  -np.cos(rrake)*np.sin(rstr)+np.cos(rdip)*np.sin(rrake)*np.cos(rstr)
        # dz = np.sin(rdip) *np.sin(rrake)

        Mxx = M0 * 2 * dx * nx
        Myy = M0 * 2 * dy * ny
        Mzz = M0 * 2 * dz * nz
        Mxy = M0 * dx * ny + dy * nx
        Mxz = M0 * dx * nz + dz * nx
        Myz = M0 * dy * nz + dz * ny

        moment = np.array([Mxx, Myy, Mzz, Mxy, Mxz, Myz])
        stations = {'names': ['S01', 'S02', 'S03', 'S04'],
                    'azimuth': np.array([120., 5., 250., 75.]),
                    'takeoff_angle': np.array([30., 60., 45., 10.]),
                    'polarity': np.array([0.8, 0.5, 0.7, -0.9])}
        # MTplot(np.array([[1], [0], [-1], [0], [0], [0]]), 'beachball',  stations=stations, fault_plane=True)
        beachball(moment, size=200, linewidth=2, facecolor='b', outfile=savepath )

    def get_stereonets(self, filepath, savename, directory, show):
        dir = directory + '/%s' % (savename.strip('.yaml'))
        if not os.path.exists(dir):
            os.makedirs(dir)

        dir_seaborn = dir + '/Stereonet'
        if not os.path.exists(dir_seaborn):
            os.makedirs(dir_seaborn)

        if filepath.endswith('.yaml') == True:
            with open(filepath, 'r') as stream:
                data_file = yaml.load(stream)
                stream.close()
                data = data_file['data']
                # parameters = data_file['parameters']
        else:
            # parameters = open(filepath, "r").readlines()[:33]
            data = np.loadtxt(filepath, delimiter=',', skiprows=70)

        df = pd.DataFrame(data,
                          columns=["Epicentral_distance", "Depth", "Strike", "Dip", "Rake", "Misfit_accepted",
                                   "Misfit_rejected", "Acceptance", "Epi_reject", "depth_reject", "Strike_reject",
                                   "Dip_reject", "Rake_reject"])
        fig, ax = mplstereonet.subplots()

        strikes = df[["Strike"]]
        dips = df[["Dip"]]

        cax = ax.density_contourf(strikes, dips, measurement='poles')

        ax.pole(strikes, dips)
        ax.grid(True)
        fig.colorbar(cax)

        plt.show()
        a=1

    def Seaborn_plots(self, filepath, savename, directory, show, skiprows, column_names):
        dir = directory + '/%s' % (savename.strip('.yaml'))
        if not os.path.exists(dir):
            os.makedirs(dir)

        dir_seaborn = dir + '/Seaborn'
        if not os.path.exists(dir_seaborn):
            os.makedirs(dir_seaborn)

        if filepath.endswith('.yaml') == True:
            with open(filepath, 'r') as stream:
                data_file = yaml.load(stream)
                stream.close()
                data = data_file['data']
                # parameters = data_file['parameters']
        else:
            # parameters = open(filepath, "r").readlines()[:33]
            data = np.loadtxt(filepath, delimiter=',', skiprows=skiprows)
        length = len(data[0]) - (len(column_names) + 3)
        R_length = 4#int(data[0][-2])
        L_length = 4#int(data[0][-1])

        for i in range(R_length):
            column_names = np.append(column_names,"R_%i" % (i+1))
        for i in range(L_length):
            column_names = np.append(column_names,"L_%i" % (i+1))
        column_names = np.append(column_names,"Accepted")
        # column_names = np.append(column_names,"Rayleigh_length")
        # column_names = np.append(column_names,"Love_length")

        df = pd.DataFrame(data,
                          columns=column_names)
        df_select = df[["Epi", "Depth", "Strike", "Dip", "Rake"]]
        # params = {'legend.fontsize': 'x-large',
        #           'figure.figsize': (15, 15),
        #           'axes.labelsize': 25,
        #           'axes.titlesize': 'x-large',
        #           'xtick.labelsize': 25,
        #           'ytick.labelsize': 25}
        # pylab.rcParams.update(params)
        # plt.figure()
        # autocorrelation_plot(df_select['Strike'],linewidth = 1,label = "Strike")
        # autocorrelation_plot(df_select['Dip'],linewidth = 1, label = "Dip")
        # autocorrelation_plot(df_select['Rake'],linewidth = 1, label = "Rake")
        # autocorrelation_plot(df_select['Epi'],linewidth = 1, label = "Epicentral_distance")
        # autocorrelation_plot(df_select['Depth'],linewidth = 1, label = "Depth")
        # plt.legend()
        # plt.savefig(directory + '/autocorrelation.pdf')
        # # plt.show()
        params = {'legend.fontsize': 'x-large',
                  'figure.figsize': (15, 15),
                  'axes.labelsize': 25,
                  'axes.titlesize': 'x-large',
                  'xtick.labelsize': 25,
                  'ytick.labelsize': 25}
        pylab.rcParams.update(params)
        # plt.figure()
        scatter_matrix(df_select, diagonal='kde')
        plt.savefig(directory + '/correlations.pdf')
        #





        #
        #

        # plot = Plots()
        # for i in itertools.combinations(df_select, 2):
        #     plot.Kernel_density(data=df_select, name_x=i[0], name_y=i[1], directory=dir_seaborn,
        #                         savename=savename.strip(".yaml"), show=show)
        #     plot.hist(data=df, name_x=i[0], name_y=i[1], directory=dir_seaborn, savename=savename.strip(".yaml"),
        #               show=show)

        ## Pair Grid approximation
        # plot.Pair_Grid(data=df_select,directory=dir_seaborn,savename=savename,show=show)

    def event_plot(self, la_receiver, lo_receiver, la_source, lo_source):
        la_r = la_receiver
        lo_r = lo_receiver
        la_s = la_source
        lo_s = lo_source
        plots = Plots()
        plots.plot_real_event(la_r, lo_r, la_s, lo_s)

    def get_seismogram_plots(self, directory, sdr=False):
        if sdr == True:
            dir = directory + '/proc_sdr'
        else:
            dir = directory + '/proc'
        filenames = glob.glob("%s/*.yaml" % dir)
        for file in filenames:
            with open(file, 'r') as stream:
                data = yaml.load(stream)
                stream.close
            f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
            for i, v in data.iteritems():
                ax1.plot(v['trace_z'], alpha=0.2)
                ax2.plot(v['trace_r'], alpha=0.2)
                ax3.plot(v['trace_t'], alpha=0.2)
            plt.show()

    def plot_streams(self,stream_filepath,filepath,savename, directory,skiprows ,column_names):
        dir = directory + '/%s' % (savename.strip('.yaml'))
        if not os.path.exists(dir):
            os.makedirs(dir)

        dir_pdf = dir + '/stream_plots'
        if not os.path.exists(dir_pdf):
            os.makedirs(dir_pdf)

        if filepath.endswith('.yaml') == True:
            with open(filepath, 'r') as stream:
                data_file = yaml.load(stream)
                stream.close()
                data = data_file['data']
                # parameters = data_file['parameters']
        else:
            # parameters = open(filepath, "r").readlines()[:33]
            data = np.loadtxt(filepath, delimiter=',', skiprows=skiprows)
        length = len(data[0]) - (len(column_names) + 3)
        R_length = int(data[0][-2])
        L_length = int(data[0][-1])

        for i in range(R_length):
            column_names = np.append(column_names,"R_%i" % (i+1))
        for i in range(L_length):
            column_names = np.append(column_names,"L_%i" % (i+1))
        column_names = np.append(column_names,"Accepted")
        column_names = np.append(column_names,"Rayleigh_length")
        column_names = np.append(column_names,"Love_length")

        st = obspy.read(stream_filepath)
        for i in st.traces:
            if "Z" in i.meta.channel:
                a=1

    def write_to_dict(self, list_of_parameters):
        parameters = {
            'velocity_model': list_of_parameters[0].strip('\n'),
            'MCMC': list_of_parameters[1].strip('\r\n'),
            'misfit': list_of_parameters[2].strip('\r\n'),
            'noise': eval(list_of_parameters[3]),
            'sdr': eval(list_of_parameters[4].strip('\n')),
            'plot_modus': eval(list_of_parameters[5].strip('\n')),
            'alpha': np.float(list_of_parameters[6]),
            'beta': np.float(list_of_parameters[7]),
            'azimuth': np.float(list_of_parameters[8]),
            'components': np.asarray(list_of_parameters[9].strip('\r\n').split(',')),
            'la_r': np.float(list_of_parameters[10]),
            'lo_r': np.float(list_of_parameters[11]),
            'filter': list_of_parameters[12].strip('\r\n'),
            'definition': list_of_parameters[13].strip('\r\n'),
            'kind': list_of_parameters[14].strip('\r\n'),
            'network': list_of_parameters[15].strip('\r\n'),
            'sample_number': np.int(list_of_parameters[16]),
            'var_est': np.float(list_of_parameters[17]),
            'epi_range_max': np.int(list_of_parameters[18]),
            'epi_range_min': np.int(list_of_parameters[19]),
            'epi_spread': np.int(list_of_parameters[20]),
            'depth_range_max': np.int(list_of_parameters[21]),
            'depth_range_min': np.int(list_of_parameters[22]),
            'depth_spread': np.int(list_of_parameters[23]),
            'strike_range_max': np.int(list_of_parameters[24]),
            'strike_range_min': np.int(list_of_parameters[25]),
            'strike_spread': np.int(list_of_parameters[26]),
            'dip_range_max': np.int(list_of_parameters[27]),
            'dip_range_min': np.int(list_of_parameters[28]),
            'dip_spread': np.int(list_of_parameters[29]),
            'rake_range_max': np.int(list_of_parameters[30]),
            'rake_range_min': np.int(list_of_parameters[31]),
            'rake_spread': np.int(list_of_parameters[32]),
            'time_at_rec': np.asarray(list_of_parameters[33].strip('\r\n').split(','), dtype=int)}
        return parameters


if __name__ == '__main__':
    main()
