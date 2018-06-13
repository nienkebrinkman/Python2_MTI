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
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
import datetime
import obspy
import matplotlib
import instaseis
import yaml

from Get_Parameters import Get_Paramters

def main():
    # Initiate Parameters:
    get_parameters = Get_Paramters()
    PRIOR = get_parameters.get_prior()
    VALUES = get_parameters.specifications()

    ## DISCUSS THIS!!!!
    PRIOR['az'] = 12.0064880807
    PRIOR['baz'] = 195.511305403

    ## Post - Processing [processing the results from inversion]
    result = Post_processing()
    file_path , savename = result.combine_parallel(dir_of_txt_files=VALUES['directory']+ '/proc_sdr',savename='new_veloc')
    result.Seaborn_plots(filepath=file_path,savename=savename,directory=VALUES['directory'])
    # result.event_plot(file_path)
    # result.get_seismogram_plots(sampler['directory'])
class Post_processing:
    def read_yaml_files(self, filepath):

        with open(filepath, 'r') as stream:
            data = yaml.load(stream)
            stream.close()
        plot = Plots()
        plot.Compare_seismograms(data['d_syn_window'], data['d_obs_window'])

    def combine_parallel(self, dir_of_txt_files,savename):
        filenames = glob.glob("%s/*.txt" % dir_of_txt_files)
        save_path = dir_of_txt_files+'/%s.yaml' % savename
        if os.path.isfile(save_path) == True:
            return save_path, savename
        for i, fname in enumerate(filenames):

            with open(fname) as infile:
                if i == 0:
                    dat = np.genfromtxt(infile, delimiter=',',skip_footer=1)
                    # par = open(fname, "r").readlines()[:34]
                    # parameters = self.write_to_dict(par)

                else:
                    # dat = np.vstack((dat, np.genfromtxt(infile, delimiter=',', skip_header=34, skip_footer=1)))
                    dat = np.vstack((dat, np.genfromtxt(infile, delimiter=',',skip_footer=1)))
            # data_file = {'data': dat, 'parameters': parameters}
            data_file = {'data': dat}
            with open(save_path, 'w') as yaml_file:
                yaml.dump(data_file, yaml_file, default_flow_style=False)
            yaml_file.close()
            return save_path, savename

    def get_pdf(self, savename, directory, filepath):
        # PDF plots will be made

        if savename.endswith('.yaml') == True:
            # data[0] - Epicentral distance [Degrees]
            # data[1] - Depth               [Meters]
            # data[2] - Time                [Min]
            # data[3] - Time                [sec]
            # data[4] - Misfit              [-]
            # data[5] - Moment_xx           [Nm]
            # data[6] - Moment_zz           [Nm]
            # data[7] - Moment_xy           [Nm]
            # data[8] - Moment_xz           [Nm]
            # data[9] - Moment_yz           [Nm]
            with open(filepath, 'r') as stream:
                data_file = yaml.load(stream)
                stream.close()

            data = np.transpose(data_file['data'])
            parameters = data_file['parameters']
            if len(data) == 10:

                time = np.array([])
                for i in range(0, len(data[2])):
                    time = np.append(time, (data[2][i] - parameters['origin_time'][4]) * 60 + data[3][i])
                data_dict = {'Epicentral_distance': data[0],
                             'Depth': data[1],
                             'Time': time,
                             'Misfit': data[4]}
            else:
                time = np.array([])
                for i in range(0, len(data[2])):
                    time = np.append(time, (data[2][i] - parameters['origin_time'][4]) * 60 + data[3][i])
                data_dict = {'Epicentral_distance': data[0],
                             'Depth': data[1],
                             'Time': time,
                             'Misfit': data[4]}

            pdf = Plots()
            for i, value in data_dict.iteritems():
                pdf.marginal_1D(data=value, name=i, amount_bins=20, directory=directory, filename=savename,
                                true_time=parameters['origin_time'])
            for i in itertools.combinations(data_dict, 2):
                pdf.marginal_2D(data_dict[i[0]], i[0], data_dict[i[1]], i[1], amount_bins=20, directory=directory,
                                filename=savename, true_time=parameters['origin_time'])



        elif savename.endswith('.txt') == True:
            parameters = open(filepath, "r").readlines()[:35]
            data = np.transpose(np.loadtxt(filepath, delimiter=',', skiprows=70))
            data_dict = {'Epicentral_distance': data[0],
                         'Depth': data[1],
                         'Time': data[2],
                         'Misfit': data[3]}

            pdf = Plots()
            for i, value in data_dict.iteritems():
                pdf.marginal_1D(data=value, name=i, amount_bins=20, directory=directory, filename=savename,
                                true_time=parameters['origin_time'])
            for i in itertools.combinations(data_dict, 2):
                pdf.marginal_2D(data_dict[i[0]], i[0], data_dict[i[1]], i[1], amount_bins=20, directory=directory,
                                filename=savename, true_time=parameters['origin_time'])

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

    def Seaborn_plots(self, filepath, savename, directory):
        dir = directory + '/Seaborn'
        if not os.path.exists(dir):
            os.makedirs(dir)

        dir_seaborn = dir + '/%s' %(savename.strip('.yaml'))
        if not os.path.exists(dir_seaborn):
            os.makedirs(dir_seaborn)

        if filepath.endswith('.yaml') == True:
            with open(filepath, 'r') as stream:
                data_file = yaml.load(stream)
                stream.close()
                data = data_file['data']
                # parameters = data_file['parameters']
        else:
            parameters = open(filepath, "r").readlines()[:33]
            data = np.transpose(np.loadtxt(filepath, delimiter=',', skiprows=33))

        # or_time = obspy.UTCDateTime(parameters['time_at_rec'][0], parameters['time_at_rec'][1],
        #                             parameters['time_at_rec'][2], parameters['time_at_rec'][3],
        #                             parameters['time_at_rec'][4], parameters['time_at_rec'][5])

        if len(data[0]) == 8:
            df = pd.DataFrame(data, columns=["Epicentral_distance", "Depth",  "Mxx", "Myy","Mxy", "Mxz", "Myz","Misfit"])
            df_select = df[['Epicentral_distance', 'Depth', 'Misfit']]
        else:
            df = pd.DataFrame(data,
                              columns=["Epicentral_distance", "Depth","Misfit", "i","j"])
            df_select = df[
                ['Epicentral_distance', 'Depth', 'Misfit']]
            # df = pd.DataFrame(data,columns=['Epicentral_distance', 'Depth', 'Misfit', 'I'])
            # df_select = df[['Epicentral_distance', 'Depth', 'Misfit']]

        plot = Plots()

        for i in itertools.combinations(df_select, 2):
            plot.Kernel_density(data=df_select, data_x=i[0], data_y=i[1], directory=dir_seaborn, savename=savename.strip(".yaml"))
            plot.hist(data=df, data_x=i[0], data_y=i[1], directory=dir_seaborn, savename=savename.strip(".yaml"))

        ## Pair Grid approximation
        plot.Pair_Grid(data=df_select,directory=dir_seaborn,savename=savename.strip(".yaml"))

    def event_plot(self, filepath):
        if filepath.endswith('.yaml') == True:
            with open(filepath, 'r') as stream:
                data_file = yaml.load(stream)
                stream.close()
                data = data_file['data']
                parameters = data_file['parameters']

        else:
            parameters = open(filepath, "r").readlines()[:36]
            data = np.transpose(np.loadtxt(filepath, delimiter=',', skiprows=36))

        la_r = parameters['la_r']
        lo_r = parameters['lo_r']
        la_s = parameters['la_s']
        lo_s = parameters['lo_s']
        plots = Plots()
        plots.plot_real_event(la_r, lo_r, la_s, lo_s)

    def get_seismogram_plots(self, directory,sdr=False):
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
