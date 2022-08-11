import argparse
import bz2
import os

import dill as pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


# Command Line Usage Examples
# Not tuning
# python3 01_construct_db.py '/scratch/gpfs/jf3375/data/scenario_data/v0.4.1-20k'
# '/scratch/gpfs/jf3375/output/scenario_data/v0.4.1-20k' 'Scenario' --energy_type 'Wind'
#
# python3 01_construct_db.py '/scratch/gpfs/jf3375/data/scenario_data/v0.4.1-20k'
# '/scratch/gpfs/jf3375/output/scenario_data/v0.4.1-20k' 'Scenario' --energy_type 'Load'
#
# python3 01_construct_db.py '/scratch/gpfs/jf3375/data/scenario_data/v0.4.1-20k'
# '/scratch/gpfs/jf3375/output/scenario_data/v0.4.1-20k' 'Scenario' --energy_type 'Solar'
#
# python3 01_construct_db.py '/scratch/gpfs/jf3375/data/scenario_data/v0.4.1-20k'
# '/scratch/gpfs/jf3375/output/score_data/v0.4.1-20k' 'Scores'
#
# Tuning
# python3 /home/jf3375/repos/PGscen/01_construct_db.py '/home/jf3375/pgscen-output/pca'
# '/home/jf3375/pgscen-output/tuning_final_files' 'Scores' --tuning-list-1 0.60 0.65 0.70 0.75 0.80 0.85  0.90 0.95 0.99  --tuning 'components'


def main():
    parser = argparse.ArgumentParser(description='Enter directories to merge and compress data files')

    parser.add_argument("inp_dir", type=str,
                        help="where the input is saved")

    parser.add_argument("out_dir", type=str,
                        help="where the final output is saved")

    parser.add_argument("data_type", type=str, choices=['Scores', 'Scenario'],
                        help="extract the data type")

    parser.add_argument("--energy_type", type=str, choices=['Wind', 'Load', 'Solar'],
                        help="extract the energy type; extract scores data will be for all energy types")

    # add arguments to parser for different tuning types
    parser.add_argument('--tuning', type=str, default='', dest='tuning',
                        choices=['rhos', 'nearest_days', 'load_specific', 'components'],
                        help='string to indicate the tuning type')
    parser.add_argument('--tuning-list-1', action="extend", nargs="+", type=float,
                        dest='tuning_list_1', help='the list of tuning param 1')
    parser.add_argument('--tuning-list-2', action="extend", nargs="+", type=float,
                        dest='tuning_list_2', help='the list of tuning param 2')

    args = parser.parse_args()

    if args.tuning == '':
        if args.data_type == 'Scenario':
            merge_output_scenarios_files_daily(args.inp_dir, args.out_dir, args.energy_type)
        else:
            merge_output_scores_files(args.inp_dir, args.out_dir)
    else:
        if args.tuning == 'rhos' or args.tuning == 'load_specific':
            if args.data_type == 'Scenario':
                Parallel(n_jobs=31, verbose=-1)(
                    delayed(merge_output_scenarios_files_daily)(args.inp_dir, args.out_dir, args.energy_type,
                                                                args.tuning, param1,
                                                                param2) for param1 in
                    args.tuning_list_1 for param2 in args.tuning_list_2)
            else:
                Parallel(n_jobs=31, verbose=-1)(
                    delayed(merge_output_scores_files)(args.inp_dir, args.out_dir, args.tuning, param1,
                                                       param2) for param1 in
                    args.tuning_list_1 for param2 in args.tuning_list_2)
        else:
            if args.data_type == 'Scenario':
                Parallel(n_jobs=31, verbose=-1)(
                    delayed(merge_output_scenarios_files_daily)(args.inp_dir, args.out_dir, args.energy_type,
                                                                args.tuning, param1) for param1 in
                    args.tuning_list_1)
            else:
                Parallel(n_jobs=31, verbose=-1)(
                    delayed(merge_output_scores_files)(args.inp_dir, args.out_dir, args.tuning, param1) for param1 in
                    args.tuning_list_1)


def extract_daily_file(file, energy_type, input_dir, drop_index=True):
    os.chdir(input_dir)
    with bz2.BZ2File(file, 'r') as f:
        data = pickle.load(f)

    if drop_index:
        daily_file = data[energy_type].stack(0).reset_index(drop=drop_index)
    else:
        daily_file = data[energy_type].stack(0).reset_index().rename(
            columns={'level_0': 'scenario', 'level_1': energy_type.lower()})
    return daily_file


def output_file(file, file_name, output_dir):
    os.chdir(output_dir)
    file.to_csv(file_name + '.csv.gz', index=False, compression='gzip')

    # with bz2.BZ2File(file_name + '.p.gz', 'w') as f:
    #     pickle.dump(file.to_dict(), f, protocol=-1)


def merge_output_scenarios_files_quarterly(input_dir, output_dir, n_jobs=30):
    # initialize output dataframes

    os.chdir(input_dir)
    files_list = sorted([i for i in os.listdir(input_dir) if i[0:5] == 'scens'])

    energy_types = ['Wind', 'Load', 'Solar']
    dates_indexes = [0, 91, 182, 274, len(files_list)]
    # dates_indexes = [0, len(files_list)]

    for energy_type in energy_types:
        outputs_list = []
        files_name_list = []
        for date_idx in range(len(dates_indexes) - 1):
            os.chdir(input_dir)
            start_date = dates_indexes[date_idx]
            end_date = dates_indexes[date_idx + 1]
            panel_df_list = Parallel(n_jobs, verbose=-1)(
                delayed(extract_daily_file)(file, energy_type) for file in files_list[start_date:end_date])
            panel_df = pd.concat(panel_df_list, axis=1)
            panel_df['scenario'] = np.repeat(np.arange(0, 10000), panel_df.shape[0] / 10000)
            panel_df[energy_type.lower()] = np.tile(np.arange(0, panel_df.shape[0] / 10000, dtype=int), 10000)
            panel_df = panel_df.set_index(['scenario', energy_type.lower()])
            outputs_list.append(panel_df)
            files_name_list.append('scenario' + '_' + energy_type.lower()
                                   + '_' + files_list[start_date][6:16] + '_' + files_list[
                                                                                    dates_indexes[date_idx + 1] - 1][
                                                                                6:16])
            # panel_df.to_csv('scenario' + '_' + energy_type.lower()
            #             + '_' + files_list[start_date][6:16] + '_' + files_list[dates_indexes[date_idx+1]-1][6:16]
            #             + '.csv.gz', index=False,
            #                 compression='gzip')

        os.chdir(output_dir)
        Parallel(max(len(outputs_list), n_jobs), verbose=-1)(
            delayed(output_file)(file, file_name, output_dir) for file, file_name in zip(outputs_list, files_name_list))


def merge_output_scenarios_files_daily(input_dir, output_dir, energy_type, tuning='', param1='', param2='', n_jobs=31):
    # ending string
    if tuning != '':
        if tuning == 'rhos' or tuning == 'load_specific':
            ending_str = tuning + '_' + str(param1) + '_' + str(param2)
        else:
            ending_str = tuning + '_' + str(param1)
        # initialize the input directory with ending str
        input_dir = os.path.join(input_dir, ending_str)

    os.chdir(input_dir)
    files_list = sorted([i for i in os.listdir(input_dir) if i[0:5] == 'scens'])

    for i in range(len(files_list) // (n_jobs - 1) + 1):
        start_file_idx = (n_jobs - 1) * i
        end_file_idx = (n_jobs - 1) * i + 30

        if end_file_idx > len(files_list):
            end_file_idx = len(files_list)

        files_name_list = ['scenario' + '_' + energy_type.lower() + '_' + i[6:16] + ending_str
                           for i in files_list[start_file_idx:end_file_idx]]

        panel_df_list = Parallel(n_jobs, verbose=-1)(
            delayed(extract_daily_file)(file, energy_type, input_dir, False) for file in
            files_list[start_file_idx:end_file_idx])

        Parallel(n_jobs, verbose=-1)(
            delayed(output_file)(file, file_name, output_dir) for file, file_name in
            zip(panel_df_list, files_name_list))


def merge_output_scores_files(input_dir, output_dir, tuning='', param1='', param2=''):
    # ending string and add it to input dir
    if tuning != '':
        if tuning == 'rhos' or tuning == 'load_specific':
            ending_str = tuning + '_' + str(param1) + '_' + str(param2)
        else:
            ending_str = tuning + '_' + str(param1)
        # initialize the input directory with ending str
        input_dir = os.path.join(input_dir, ending_str)

    os.chdir(input_dir)
    # list input fuiles
    escores_files_list = [i for i in os.listdir() if i[:7] == 'escores']
    varios_files_list = [i for i in os.listdir() if i[:6] == 'varios']

    input_files = {'escores': sorted(escores_files_list), 'varios': sorted(varios_files_list)}

    # construct output files
    for scores, files_list in input_files.items():
        os.chdir(input_dir)
        # initlizae output dataframes
        df_panel_solar = pd.DataFrame()
        df_panel_load = pd.DataFrame()
        df_panel_wind = pd.DataFrame()

        df_energy_types = {'Solar': df_panel_solar, 'Load': df_panel_load, 'Wind': df_panel_wind}
        for file in files_list:
            with bz2.BZ2File(file, 'r') as f:
                df = pd.DataFrame(pickle.load(f))
            if scores == 'escores':
                date = file[8:18]
            if scores == 'varios':
                date = file[7:17]
            for energy_type in df.columns:
                df_energy_types[energy_type] = pd.concat([df_energy_types[energy_type],
                                                          pd.Series(df[energy_type].dropna(), name=date)], axis=1)

        os.chdir(output_dir)
        for energy_type, df in df_energy_types.items():
            if df.shape[0] != 0:
                df = df.reset_index().rename(columns={'index': energy_type.lower()}).sort_values([energy_type.lower()])
                df.to_csv(scores + '_' + energy_type.lower() + '_2020_' + ending_str + '.csv.gz', index=False,
                          compression='gzip')


if __name__ == '__main__':
    main()
