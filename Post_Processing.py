# ----------------------------------------------------------------------------------------------------------------------#
#                                                 Post Processing                                                      #
# ---------------------------------------------------------------------------------------------------------------------#
from Plots import Plots
import os
import numpy as np
import itertools
import glob
import pandas as pd
import matplotlib.pylab as plt
import obspy
import matplotlib
import instaseis
import yaml


class Post_processing:
    def read_yaml_files(self, filepath):

        with open(filepath, 'r') as stream:
            data = yaml.load(stream)
            stream.close()
        plot = Plots()
        plot.Compare_seismograms(data['d_syn_window'], data['d_obs_window'])

    def combine_parallel(self, savename, directory):
        dir = directory + '/proc'
        filenames = glob.glob("%s/*.txt" % dir)
        filename = savename + '.yaml'
        save_path = directory + '/%s' % filename

        if os.path.isfile(save_path) == True:
            return save_path, filename
        else:
            for i, fname in enumerate(filenames):

                with open(fname) as infile:
                    if i == 0:
                        dat = np.genfromtxt(infile, delimiter=',', skip_header=70, skip_footer=1)
                        par = open(fname, "r").readlines()[:35]
                        parameters = self.write_to_dict(par)
                        a = 1
                    else:
                        dat = np.vstack((dat, np.genfromtxt(infile, delimiter=',', skip_header=70, skip_footer=1)))
            data_file = {'data': dat, 'parameters': parameters}
            with open(save_path, 'w') as yaml_file:
                yaml.dump(data_file, yaml_file, default_flow_style=False)
            yaml_file.close()
            return save_path, filename



    def get_pdf(self, savename, directory, filepath):
        # PDF plots will be made

        if savename.endswith('.yaml') == True:
            # data[0] - Epicentral distance [Degrees]
            # data[1] - Depth               [Meters]
            # data[2] - Time                [Seconds]
            # data[3] - Misfit              [-]
            # data[4] - Moment_xx           [Nm]
            # data[5] - Moment_zz           [Nm]
            # data[6] - Moment_xy           [Nm]
            # data[7] - Moment_xz           [Nm]
            # data[8] - Moment_yz           [Nm]
            with open(filepath, 'r') as stream:
                data_file = yaml.load(stream)
                stream.close()

            data = np.transpose(data_file['data'])

            data_dict = {'Epicentral_distance': data[0],
                         'Depth': data[1],
                         'Time': data[2],
                         'Misfit': data[3]}

            # times = matplotlib.dates.num2date(data_dict['Time'])
            # mean_time = (min(data_dict['Time']) + max(data_dict['Time']) ) / 2
            # time=obspy.UTCDateTime(mean_time)

            pdf = Plots()
            for i, value in data_dict.iteritems():
                pdf.marginal_1D(data=value, name=i, amount_bins=20, directory=directory, filename=savename)
            for i in itertools.combinations(data_dict, 2):
                pdf.marginal_2D(data_dict[i[0]], i[0], data_dict[i[1]], i[1], amount_bins=20, directory=directory,
                                filename=savename)

        elif savename.endswith('.txt') == True:
            parameters = open(filepath, "r").readlines()[:35]
            data = np.transpose(np.loadtxt(filepath, delimiter=',', skiprows=70))
            data_dict = {'Epicentral_distance': data[0],
                         'Depth': data[1],
                         'Time': data[2],
                         'Misfit': data[3]}

            pdf = Plots()
            for i, value in data_dict.iteritems():
                pdf.marginal_1D(data=value, name=i, amount_bins=20, directory=directory, filename=savename)
            for i in itertools.combinations(data_dict, 2):
                pdf.marginal_2D(data_dict[i[0]], i[0], data_dict[i[1]], i[1], amount_bins=20, directory=directory,
                                filename=savename)

        else:
            print("The file does not exist yet [FIRST RUN THE MH_ALOGRITHM!]")

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
            parameters = open(filepath, "r").readlines()[:35]
            data = np.transpose(np.loadtxt(filepath, delimiter=',', skiprows=70))

        data = np.transpose(data)
        moment_data = np.transpose(np.array([data[4], data[5], data[6], data[7], data[8]]))
        plot = Plots()
        for i in moment_data:
            plot.Beachball(i)

    def Seaborn_plots(self, filepath, savename, directory):
        dir_seaborn = directory + '/Seaborn'
        if not os.path.exists(dir_seaborn):
            os.makedirs(dir_seaborn)

        if filepath.endswith('.yaml') == True:
            with open(filepath, 'r') as stream:
                data_file = yaml.load(stream)
                stream.close()
                data = data_file['data']
                parameters = data_file['parameters']
        else:
            parameters = open(filepath, "r").readlines()[:35]
            data = np.transpose(np.loadtxt(filepath, delimiter=',', skiprows=70))
        df = pd.DataFrame(data,
                          columns=["Epicentral_distance", "Depth", "Time", "Misfit", "Mxx", "Myy", "Mxy", "Mxz", "Myz"])
        df_select = df[['Epicentral_distance', 'Depth', 'Time', 'Misfit']]

        plot = Plots()

        ## Kernel Density approximation
        plot.Kernel_density(data=df, data_x="Epicentral_distance", data_y="Depth", parameters=parameters,
                            directory=dir_seaborn, savename=savename.strip(".yaml"))

        ## Pair Grid approximation
        # plot.Pair_Grid(data=df_select,directory=directory,savename=savename)

    def event_plot(self,filepath):
        if filepath.endswith('.yaml') == True:
            with open(filepath, 'r') as stream:
                data_file = yaml.load(stream)
                stream.close()
                data = data_file['data']
                parameters = data_file['parameters']

        else:
            parameters = open(filepath, "r").readlines()[:35]
            data = np.transpose(np.loadtxt(filepath, delimiter=',', skiprows=70))
        la_r = parameters['la_r']
        lo_r = parameters['lo_r']
        la_s = parameters['la_s']
        lo_s = parameters['lo_s']
        plots = Plots()
        plots.plot_real_event(la_r,lo_r,la_s,lo_s)


    def get_seismogram_plots(self,directory):
        dir = directory + '/proc'
        filenames = glob.glob("%s/*.yaml" % dir)
        for file in filenames:
            with open(file,'r') as stream:
                data = yaml.load(stream)
                stream.close
            a=1
            f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
            for i,v in data.iteritems():
                ax1.plot(v['trace_z'], alpha=0.2)
                ax2.plot(v['trace_r'], alpha=0.2)
                ax3.plot(v['trace_t'], alpha=0.2)
            plt.show()





    def write_to_dict(self, list_of_parameters):
        parameters = {
            'velocity_model': list_of_parameters[0].strip('\n'),
            'alpha': np.float(list_of_parameters[1]),
            'beta': np.float(list_of_parameters[2]),
            'azimuth': np.float(list_of_parameters[3]),
            'Depth': np.int(list_of_parameters[4]),
            'Epicentral_distance': np.float(list_of_parameters[5]),
            'components': np.asarray(list_of_parameters[6].strip('\r\n').split(',')),
            'la_r': np.float(list_of_parameters[7]),
            'la_s': np.float(list_of_parameters[8]),
            'lo_s': np.float(list_of_parameters[9]),
            'lo_r': np.float(list_of_parameters[10]),
            'm_pp': np.float(list_of_parameters[11]),
            'm_rp': np.float(list_of_parameters[12]),
            'm_rr': np.float(list_of_parameters[13]),
            'm_rt': np.float(list_of_parameters[14]),
            'm_tp': np.float(list_of_parameters[15]),
            'm_tt': np.float(list_of_parameters[16]),
            'strike': np.int(list_of_parameters[17]),
            'rake': np.int(list_of_parameters[18]),
            'dip': np.int(list_of_parameters[19]),
            'filter': list_of_parameters[20].strip('\r\n'),
            'definition': list_of_parameters[21].strip('\r\n'),
            'kind': list_of_parameters[22].strip('\r\n'),
            'network': list_of_parameters[23].strip('\r\n'),
            'filename': list_of_parameters[24].strip('\r\n'),
            'directory': list_of_parameters[25].strip('\r\n'),
            'filepath': list_of_parameters[26].strip('\r\n'),
            'sample_number': np.int(list_of_parameters[27]),
            'var_est': np.float(list_of_parameters[28]),
            'epi_range_max': np.int(list_of_parameters[29]),
            'epi_range_min': np.int(list_of_parameters[30]),
            'epi_step': np.int(list_of_parameters[31]),
            'depth_range_max': np.int(list_of_parameters[32]),
            'depth_range_min': np.int(list_of_parameters[33]),
            'depth_step': np.int(list_of_parameters[34])}
        return parameters
