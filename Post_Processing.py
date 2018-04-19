#----------------------------------------------------------------------------------------------------------------------#
#                                                 Post Processing                                                      #
# ---------------------------------------------------------------------------------------------------------------------#
from Plots import Plots
import os
import numpy as np
import itertools
import glob


class Post_processing:
    def __init__(self,sampler):
        self.sampler = sampler

    def combine_parallel(self):
        directory = self.sampler['directory'] + '/proc'
        filenames = glob.glob("%s/*.txt" % directory)
        filename='combined_files.txt'
        save_path = self.sampler['directory'] + '/%s' % filename
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
            # data[1] - Azimuth             [Degrees]
            # data[2] - Depth               [Meters]
            # data[3] - Time                [Seconds]
            # data[4] - Misfit              [-]
            # data[5] - Moment_xx           [Nm]
            # data[6] - Moment_zz           [Nm]
            # data[7] - Moment_xy           [Nm]
            # data[8] - Moment_xz           [Nm]
            # data[9] - Moment_yz           [Nm]

            data = np.transpose(np.loadtxt(filepath, delimiter=','))
            data_dict = {'Epicentral_distance' : data[0],
                         'Azimuth' : data[1],
                         'Depth' : data[2],
                         'Time': data[3],
                         'Misfit': data[4]}

            # with open(self.filepath, 'r') as stream:
            #     data = yaml.load(stream)
            #     stream.close()
            pdf = Plots()
            for i in itertools.combinations(data_dict, 2):
                pdf.marginal_2D(data_dict[i[0]], i[0], data_dict[i[1]], i[1], amount_bins=20, directory = directory, filename=filename)

        else:
            print("The file does not exist yet [FIRST RUN THE MH_ALOGRITHM!]")


        # TODO - Plot the results of Moment Tensor


