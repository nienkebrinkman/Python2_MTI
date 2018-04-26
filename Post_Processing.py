#----------------------------------------------------------------------------------------------------------------------#
#                                                 Post Processing                                                      #
# ---------------------------------------------------------------------------------------------------------------------#
from Plots import Plots
import os
import numpy as np
import itertools
import glob
import pandas as pd



class Post_processing:
    def __init__(self,directory):
        self.directory = directory

    def combine_parallel(self,savename):
        directory = self.directory + '/proc'
        filenames = glob.glob("%s/*.txt" % directory)
        filename= savename
        save_path = self.directory + '/%s' % filename
        if os.path.isfile(save_path) == True:
            return save_path, filename
        else:
            with open(save_path, 'w') as outfile:
                for fname in filenames:
                    with open(fname) as infile:
                        for line in infile:
                            outfile.write(line)
            return save_path, filename


    def get_pdf(self, filepath, filename, directory):
        # PDF plots will be made

        if os.path.isfile(filepath) == True:
            # data[0] - Epicentral distance [Degrees]
            # data[1] - Depth               [Meters]
            # data[2] - Time                [Seconds]
            # data[3] - Misfit              [-]
            # data[4] - Moment_xx           [Nm]
            # data[5] - Moment_zz           [Nm]
            # data[6] - Moment_xy           [Nm]
            # data[7] - Moment_xz           [Nm]
            # data[8] - Moment_yz           [Nm]

            data = np.transpose(np.loadtxt(filepath, delimiter=','))
            data_dict = {'Epicentral_distance' : data[0],
                         'Depth' : data[1],
                         'Time' : data[2],
                         'Misfit': data[3]}
            # with open(self.filepath, 'r') as stream:
            #     data = yaml.load(stream)
            #     stream.close()
            pdf = Plots()
            for i,value in data_dict.iteritems():
                pdf.marginal_1D(data=value,name=i,amount_bins=20 ,directory = directory, filename=filename)
            for i in itertools.combinations(data_dict, 2):
                pdf.marginal_2D(data_dict[i[0]], i[0], data_dict[i[1]], i[1], amount_bins=20, directory = directory, filename=filename)


        else:
            print("The file does not exist yet [FIRST RUN THE MH_ALOGRITHM!]")
    def get_beachballs(self,filepath, filename, directory):
        if os.path.isfile(filepath) == True:
            # data[0] - Epicentral distance [Degrees]
            # data[1] - Depth               [Meters]
            # data[2] - Time                [Seconds]
            # data[3] - Misfit              [-]
            # data[4] - Moment_xx           [Nm]
            # data[5] - Moment_zz           [Nm]
            # data[6] - Moment_xy           [Nm]
            # data[7] - Moment_xz           [Nm]
            # data[8] - Moment_yz           [Nm]

            data = np.transpose(np.loadtxt(filepath, delimiter=','))
            moment_data = np.transpose(np.array([data[4],data[5],data[6],data[7],data[8]]))
            plot=Plots()
            number = 0
            for i in moment_data:
                plot.Beachball(i)

    def Seaborn_plots(self,filepath, savename, directory):
        dir_seaborn = directory + '/Seaborn'
        if not os.path.exists(dir_seaborn):
            os.makedirs(dir_seaborn)


        data = np.loadtxt(filepath, delimiter=',')
        df = pd.DataFrame(data,
                          columns=["Epicentral_distance", "Depth", "Time", "Misfit", "Mxx", "Myy", "Mxy", "Mxz", "Myz"])
        df_select = df[['Epicentral_distance', 'Depth', 'Time', 'Misfit']]

        plot=Plots()

        ## Kernel Density approximation
        plot.Kernel_density(data=df,data_x="Epicentral_distance",data_y="Depth",directory=dir_seaborn,savename=savename)

        ## Pair Grid approximation
        plot.Pair_Grid(data=df_select,directory=directory,savename=savename)

