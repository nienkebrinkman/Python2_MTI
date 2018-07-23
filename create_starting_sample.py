import numpy as np
import pandas as pd

def main():

    txt_filepath = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/Blindtest/fixed_depth_epi/Exploration/Blindtest_trialrun.txt'

    create = create_starting_sample()
    create.get_sample(txt_filepath)


class create_starting_sample:
    def get_sample_manual(self,epi,depth,strike,dip,rake,savepath):
        with open(savepath, 'w') as save_file:
            save_file.write("%.4f,%.4f,%.4f,%.4f,%.4f\n\r" % (epi,depth,strike,dip,rake))
        save_file.close()
        print("Txt file with new sample is saved as %s" % savepath)
        return savepath

    def get_sample(self, txt_filepath):
        data = np.loadtxt(txt_filepath, delimiter=',', skiprows=70)
        column_names = ["Epi", "Depth", "Strike", "Dip", "Rake", "Total_misfit", "S_z", "S_r", "S_t", "P_z", "P_r",
                        "BW_misfit", "Rtot", "Ltot"]
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
        min_misfit_index = np.argmin(df['Total_misfit'], axis=None)
        max_misfit = df['Total_misfit'][min_misfit_index]

        epi = df["Epi"][min_misfit_index]
        depth = df["Depth"][min_misfit_index]
        strike = df["Strike"][min_misfit_index]
        dip = df["Dip"][min_misfit_index]
        rake = df["Rake"][min_misfit_index]

        savepath = txt_filepath.replace(".txt","_sample.txt")

        with open(savepath, 'w') as save_file:
            save_file.write("%.4f,%.4f,%.4f,%.4f,%.4f\n\r" % (epi,depth,strike,dip,rake))
        save_file.close()
        print("Txt file with new sample is saved as %s" % savepath)


if __name__ == '__main__':
    main()
