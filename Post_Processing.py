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


def main():
    ## Post - Processing [processing the results from inversion]
    result = Post_processing_sdr()

    directory = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/Additional_scripts/Iteration_runs'
    path_to_file = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/Additional_scripts/Iteration_runs/iter_4_10000.txt'

    savename = '10000'
    show = False  # Choose True for direct show, choose False for saving

    # result.get_stereonets(filepath=path_to_file, savename=savename, directory=directory, show=show)
    result.get_accepted_samples(filepath=path_to_file,savename=savename,directory=directory)
    result.get_convergence(filepath=path_to_file, savename=savename, directory=directory, show=show)
    result.Seaborn_plots(filepath=path_to_file, savename=savename, directory=directory, show=show)
    result.get_pdf(filepath=path_to_file, savename=savename, directory=directory, show=show)


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

    def get_accepted_samples(self,filepath,savename,directory):
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
            data = np.loadtxt(filepath, delimiter=',', skiprows=70)

        df = pd.DataFrame(data,
                          columns=["Epicentral_distance", "Depth", "Strike", "Dip", "Rake", "Misfit_accepted",
                                   "Misfit_rejected", "Acceptance", "Epi_reject", "depth_reject", "Strike_reject",
                                   "Dip_reject", "Rake_reject"])
        total = len(df["Acceptance"])
        accepted = np.sum(df["Acceptance"]==1)
        savepath = dir + '/Accept_%i_outof_%i.txt' % (accepted,total)
        with open(savepath, 'w') as save_file:
            save_file.write("%i,%i\n\r" % (total,accepted))
        save_file.close()
        print("Total amount of samples = %i" % len(df["Acceptance"]))
        print("Total amount of samples = %i" % np.sum(df["Acceptance"]==1))

    def get_convergence(self, filepath, savename, directory, show=True):
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
            data = np.loadtxt(filepath, delimiter=',', skiprows=70)

        df = pd.DataFrame(data,
                          columns=["Epicentral_distance", "Depth", "Strike", "Dip", "Rake", "Misfit_accepted",
                                   "Misfit_rejected", "Acceptance", "Epi_reject", "depth_reject", "Strike_reject",
                                   "Dip_reject", "Rake_reject"])

        plt.figure(1)
        ax = plt.subplot(111)
        ax.plot(np.arange(0, len(df['Misfit_accepted'])), df['Misfit_accepted'])
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

    def get_pdf(self, filepath, savename, directory, show=True):
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
            data = np.loadtxt(filepath, delimiter=',', skiprows=70)

        df = pd.DataFrame(data,
                          columns=["Epicentral_distance", "Depth", "Strike", "Dip", "Rake", "Misfit_accepted",
                                   "Misfit_rejected", "Acceptance", "Epi_reject", "depth_reject", "Strike_reject",
                                   "Dip_reject", "Rake_reject"])
        df_select = df[["Epicentral_distance", "Depth", "Strike", "Dip", "Rake"]]
        pdf = Plots()
        for i, value in df_select.iteritems():
            if i == 'Misfit_accepted' or i == 'Misfit_rejected' or i == 'Acceptance':
                continue
            else:
                pdf.marginal_1D(data=value, name=i, amount_bins=20, directory=dir_pdf, show=show)
        for i in itertools.combinations(df_select, 2):
            pdf.marginal_2D(df_select[i[0]], i[0], df_select[i[1]], i[1], amount_bins=20, directory=dir_pdf, show=show)

    def get_beachballs(self, filepath, filename, directory):

        # data[0] - Epicentral distance [Degrees]
        # data[1] - Depth               [Meters]
        # data[2] - Time                [Seconds]
        # data[3] - Misfit              [-]
        # data[4] - Moment_xx           [Nm]
        # data[5] - Moment_zz           [Nm]
        # data[6] - Moment_xy           [Nm]
        # data[7] - Moment_xz           [Nm]
        # data[8] - Moment_yz           [Nm]

        if filepath.endswith('.yaml') == True:
            with open(filepath, 'r') as stream:
                data_file = yaml.load(stream)
                stream.close()
                data = data_file['data']
                parameters = data_file['parameters']

        else:
            parameters = open(filepath, "r").readlines()[:36]
            data = np.transpose(np.loadtxt(filepath, delimiter=',', skiprows=36))
        if len(data) == 10:
            data = np.transpose(data)

            moment_data = np.transpose(np.array([data[4], data[5], data[6], data[7], data[8]]))
            plot = Plots()
            for i in moment_data:
                plot.Beachball(i)
        else:
            print('This data does not contain an inversion for moment (probably sdr = True)')

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

    def Seaborn_plots(self, filepath, savename, directory, show):
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
            data = np.loadtxt(filepath, delimiter=',', skiprows=70)

        df = pd.DataFrame(data,
                          columns=["Epicentral_distance", "Depth", "Strike", "Dip", "Rake", "Misfit_accepted",
                                   "Misfit_rejected", "Acceptance", "Epi_reject", "depth_reject", "Strike_reject",
                                   "Dip_reject", "Rake_reject"])
        df_select = df[["Epicentral_distance", "Depth", "Strike", "Dip", "Rake", "Misfit_accepted"]]

        plot = Plots()
        for i in itertools.combinations(df_select, 2):
            plot.Kernel_density(data=df_select, name_x=i[0], name_y=i[1], directory=dir_seaborn,
                                savename=savename.strip(".yaml"), show=show)
            plot.hist(data=df, name_x=i[0], name_y=i[1], directory=dir_seaborn, savename=savename.strip(".yaml"),
                      show=show)
            #
            # ## Pair Grid approximation
            # plot.Pair_Grid(data=df_select,directory=dir_seaborn,savename=savename.strip(".yaml"))

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
