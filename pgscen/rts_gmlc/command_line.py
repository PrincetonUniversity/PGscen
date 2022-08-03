"""Command line interface for generating scenarios for the RTS-GMLC system."""

import argparse
from pathlib import Path
import shutil
import time
import bz2
import dill as pickle

import numpy as np
import pandas as pd

from ..command_line import parent_parser
from ..engine import GeminiEngine, SolarGeminiEngine
from ..scoring import compute_energy_scores, compute_variograms

from .data_utils import load_load_data, load_wind_data, load_solar_data
from ..utils.data_utils import (split_actuals_hist_future,
                                split_forecasts_hist_future)


rts_parser = argparse.ArgumentParser(add_help=False, parents=[parent_parser])
rts_parser.add_argument('rts_dir', type=str,
                        help="where RTS-GMLC repository is stored")

joint_parser = argparse.ArgumentParser(parents=[rts_parser], add_help=False)
joint_parser.add_argument('--use-all-load-history',
                          action='store_true', dest='use_all_load_hist',
                          help="train load models using all out-of-sample "
                               "historical days instead of the same "
                               "window used for solar models")


#TODO: RTS solar models seem to be numerically unstable — why?
def run_rts():
    args = argparse.ArgumentParser(
        'pgscen-rts', parents=[rts_parser],
        description="Create day-ahead RTS load, wind, and solar scenarios."
        ).parse_args()

    rts_runner(args.start, args.days, args.rts_dir, args.out_dir,
               args.scenario_count, args.nearest_days, args.asset_rho,
               args.time_rho, args.random_seed, create_load=True,
               create_wind=True, create_solar=True, write_csv=not args.pickle,
               skip_existing=args.skip_existing,
               get_energy_scores=args.energy_scores,
               get_variograms=args.variograms, verbosity=args.verbose)

def run_rts_solar():
    args = argparse.ArgumentParser(
        'pgscen-rts-solar', parents=[rts_parser],
        description="Create day-ahead RTS solar scenarios."
    ).parse_args()

    rts_runner(args.start, args.days, args.rts_dir, args.out_dir,
               args.scenario_count, args.nearest_days, args.asset_rho,
               args.time_rho, args.random_seed, create_load=False,
               create_wind=False, create_solar=True, write_csv=not args.pickle,
               skip_existing=args.skip_existing,
               get_energy_scores=args.energy_scores,
               get_variograms=args.variograms, verbosity=args.verbose)

def run_rts_joint():
    args = argparse.ArgumentParser(
        'pgscen-rts-joint', parents=[joint_parser],
        description="Create day-ahead RTS wind and load-solar joint scenarios."
        ).parse_args()

    rts_runner(args.start, args.days, args.rts_dir, args.out_dir,
               args.scenario_count, args.nearest_days, args.asset_rho,
               args.time_rho, args.random_seed, create_load=False,
               create_wind=True, create_load_solar=True,
               write_csv=not args.pickle, skip_existing=args.skip_existing,
               use_all_load_hist=args.use_all_load_hist,
               get_energy_scores=args.energy_scores,
               get_variograms=args.variograms, verbosity=args.verbose)


def rts_runner(start_date, ndays, rts_dir, out_dir, scen_count, nearest_days,
               asset_rho, time_rho, random_seed, create_load=False,
               create_wind=False, create_solar=False, create_load_solar=False,
               write_csv=True, skip_existing=False, use_all_load_hist=False,
               get_energy_scores=False, get_variograms=False, tuning=False, verbosity=0):
    start = ' '.join([start_date, "08:00:00"])

    if random_seed:
        np.random.seed(random_seed)

    if create_load or create_load_solar:
        load_zone_actual_df, load_zone_forecast_df = load_load_data(rts_dir)

    if create_wind:
        (wind_site_actual_df, wind_site_forecast_df,
            wind_meta_df) = load_wind_data(rts_dir)

    if create_solar or create_load_solar:
        (solar_site_actual_df, solar_site_forecast_df,
            solar_meta_df) = load_solar_data(rts_dir)

    if verbosity >= 2:
        t0 = time.time()

    for scenario_start_time in pd.date_range(start=start, periods=ndays,
                                             freq='D', tz='utc'):
        date_lbl = scenario_start_time.strftime('%Y-%m-%d')

        scen_timesteps = pd.date_range(start=scenario_start_time,
                                       periods=24, freq='H')

        if verbosity >= 1:
            print("Creating RTS-GMLC scenarios for: {} with {} {}".format(date_lbl, asset_rho, time_rho))

        if get_energy_scores:
            energy_scores = dict()
        if get_variograms:
            variograms = dict()

        if not write_csv and not tuning:
            out_fl = Path(out_dir, "scens_{}_{}_{}.p.gz".format(date_lbl, asset_rho, time_rho))

            if skip_existing and out_fl.exists():
                continue

            out_scens = dict()

        if write_csv and skip_existing and not tuning:
            date_path = Path(out_dir, scenario_start_time.strftime('%Y%m%d'))

            if ((create_load or create_load_solar)
                    and Path(date_path, 'load').exists()):
                shutil.rmtree(Path(date_path, 'load'))

            if create_wind and Path(date_path, 'wind').exists():
                shutil.rmtree(Path(date_path, 'wind'))

            if ((create_solar or create_load_solar)
                    and Path(date_path, 'solar').exists()):
                shutil.rmtree(Path(date_path, 'solar'))

        # split input datasets into training and testing subsets
        # for RTS we always do in-sample since we only have a year of data
        if create_load or create_load_solar:
            (load_zone_actual_hists,
                load_zone_actual_futures) = split_actuals_hist_future(
                        load_zone_actual_df, scen_timesteps, in_sample=True)
            (load_zone_forecast_hists,
                load_zone_forecast_futures) = split_forecasts_hist_future(
                        load_zone_forecast_df, scen_timesteps, in_sample=True)

        if create_wind:
            (wind_site_actual_hists,
                wind_site_actual_futures) = split_actuals_hist_future(
                        wind_site_actual_df, scen_timesteps, in_sample=True)
            (wind_site_forecast_hists,
                wind_site_forecast_futures) = split_forecasts_hist_future(
                        wind_site_forecast_df, scen_timesteps, in_sample=True)

            wind_engn = GeminiEngine(wind_site_actual_hists,
                                     wind_site_forecast_hists,
                                     scenario_start_time, wind_meta_df, 'wind')

        if create_solar or create_load_solar:
            (solar_site_actual_hists,
                solar_site_actual_futures) = split_actuals_hist_future(
                        solar_site_actual_df, scen_timesteps, in_sample=True)
            (solar_site_forecast_hists,
                solar_site_forecast_futures) = split_forecasts_hist_future(
                        solar_site_forecast_df, scen_timesteps, in_sample=True)

            solar_engn = SolarGeminiEngine(solar_site_actual_hists,
                                           solar_site_forecast_hists,
                                           scenario_start_time, solar_meta_df,
                                           us_state='California')

        if create_load:
            load_engn = GeminiEngine(load_zone_actual_hists,
                                     load_zone_forecast_hists,
                                     scenario_start_time, asset_type='load')

            load_engn.fit(asset_rho, time_rho, nearest_days)
            load_engn.create_scenario(scen_count, load_zone_forecast_futures,
                                      bin_width_ratio=0.1, min_sample_size=400)

            if get_energy_scores:
                energy_scores['Load'] = compute_energy_scores(
                    load_engn.scenarios['load'],
                    load_zone_actual_df, load_zone_forecast_df
                )

            if get_variograms:
                variograms['Load'] = compute_variograms(
                    load_engn.scenarios['load'],
                    load_zone_actual_df, load_zone_forecast_df
                )
            if not tuning:
                if write_csv:
                    load_engn.write_to_csv(out_dir, load_zone_actual_futures,
                                           write_forecasts=True)
                else:
                    out_scens['Load'] = load_engn.scenarios['load'].round(4)

        if create_wind:
            dist = wind_engn.asset_distance().values
            wind_engn.fit(2 * asset_rho * dist / dist.max(), time_rho,
                          nearest_days)
            wind_engn.create_scenario(scen_count, wind_site_forecast_futures)

            if get_energy_scores:
                energy_scores['Wind'] = compute_energy_scores(
                    wind_engn.scenarios['wind'],
                    wind_site_actual_df, wind_site_forecast_df
                )

            if get_variograms:
                variograms['Wind'] = compute_variograms(
                    wind_engn.scenarios['wind'],
                    wind_site_actual_df, wind_site_forecast_df
                )

            if not tuning:
                if write_csv:
                    wind_engn.write_to_csv(out_dir, wind_site_actual_futures,
                                           write_forecasts=True)
                else:
                    out_scens['Wind'] = wind_engn.scenarios['wind'].round(4)

        if create_solar:
            solar_engn.fit_solar_model(nearest_days=nearest_days)
            solar_engn.create_solar_scenario(scen_count,
                                             solar_site_forecast_futures)

            if get_energy_scores:
                energy_scores['Solar'] = compute_energy_scores(
                    solar_engn.scenarios['solar'],
                    solar_site_actual_df, solar_site_forecast_df
                )

            if get_variograms:
                variograms['Solar'] = compute_variograms(
                    solar_engn.scenarios['solar'],
                    solar_site_actual_df, solar_site_forecast_df
                )
            if not tuning:
                if write_csv:
                    solar_engn.write_to_csv(out_dir,
                                            {'solar': solar_site_actual_futures},
                                            write_forecasts=True)
                else:
                    out_scens['Solar'] = solar_engn.scenarios['solar'].round(4)

        if create_load_solar:
            solar_engn.fit_load_solar_joint_model(
                load_zone_actual_hists, load_zone_forecast_hists,
                nearest_days=nearest_days, use_all_load_hist=use_all_load_hist
                )

            solar_engn.create_load_solar_joint_scenario(
                scen_count,
                load_zone_forecast_futures, solar_site_forecast_futures
                )

            if get_energy_scores:
                energy_scores['Load'] = compute_energy_scores(
                    solar_engn.scenarios['load'],
                    load_zone_actual_df, load_zone_forecast_df
                    )
                energy_scores['Solar'] = compute_energy_scores(
                    solar_engn.scenarios['solar'],
                    solar_site_actual_df, solar_site_forecast_df
                    )

            if get_variograms:
                variograms['Load'] = compute_variograms(
                    solar_engn.scenarios['load'],
                    load_zone_actual_df, load_zone_forecast_df
                )
                variograms['Solar'] = compute_variograms(
                    solar_engn.scenarios['solar'],
                    solar_site_actual_df, solar_site_forecast_df
                )

            if not tuning:
                if write_csv:
                    solar_engn.write_to_csv(out_dir,
                                            {'load': load_zone_actual_futures,
                                             'solar': solar_site_actual_futures},
                                            write_forecasts=True)
                else:
                    out_scens['Load'] = solar_engn.scenarios['load'].round(4)
                    out_scens['Solar'] = solar_engn.scenarios['solar'].round(4)

        if not write_csv and not tuning:
            with bz2.BZ2File(out_fl, 'w') as f:
                pickle.dump(out_scens, f, protocol=-1)

        if get_energy_scores:
            with bz2.BZ2File(Path(out_dir,
                                  'escores_{}_{}_{}.p.gz'.format(date_lbl, asset_rho, time_rho)),
                             'w') as f:
                pickle.dump(energy_scores, f, protocol=-1)

        if get_variograms:
            with bz2.BZ2File(Path(out_dir,
                                  'varios_{}_{}_{}.p.gz'.format(date_lbl, asset_rho, time_rho)),
                             'w') as f:
                pickle.dump(variograms, f, protocol=-1)

    if verbosity >= 2:
        if ndays == 1:
            day_str = "day"
        else:
            day_str = "days"

        type_lbls = list()
        if create_load:
            type_lbls += ['load']
        if create_wind:
            type_lbls += ['wind']
        if create_solar:
            type_lbls += ['solar']
        if create_load_solar:
            type_lbls += ['joint load-solar']

        if len(type_lbls) == 1:
            type_str = type_lbls[0]
        elif len(type_lbls) == 2:
            type_str = ' and '.join(type_lbls)

        else:
            type_str = ', and '.join([', '.join(type_lbls[:-1]),
                                      type_lbls[-1]])

        print("Created {} {} scenarios for {} {} in {:.1f} seconds".format(
            scen_count, type_str, ndays, day_str, time.time() - t0))
