import numpy as np
import pandas as pd

def main():

    txt_filepath = '/home/nienke/Documents/Applied_geophysics/Thesis/anaconda/Additional_scripts/Iteration_runs/iter_4_10000.txt'

    create = create_starting_sample()
    create.get_sample(txt_filepath)


class create_starting_sample:

    def get_sample(self, txt_filepath):
        data = np.loadtxt(txt_filepath, delimiter=',', skiprows=70)

        df = pd.DataFrame(data,
                          columns=["Epicentral_distance", "Depth", "Strike", "Dip", "Rake", "Misfit_accepted",
                                   "Misfit_rejected", "Acceptance", "Epi_reject", "depth_reject", "Strike_reject",
                                   "Dip_reject", "Rake_reject"])
        min_misfit_index = np.argmin(df['Misfit_accepted'], axis=None)

        epi = df["Epicentral_distance"][min_misfit_index]
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
