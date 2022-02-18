"""Command line interface for generating scenarios with ERCOT/NREL datasets."""

import argparse
import os
import pandas as pd
from pathlib import Path
import bz2
import dill as pickle
import time

from .utils.data_utils import (load_load_data, load_wind_data, load_solar_data,
                               split_actuals_hist_future,
                               split_forecasts_hist_future)
from .engine import GeminiEngine, SolarGeminiEngine


parent_parser = argparse.ArgumentParser(add_help=False)

parent_parser.add_argument(
    'start', type=str,
    help="start date for the scenarios, given in YYYY-MM-DD format"
    )
parent_parser.add_argument(
    'days', type=int, help="for how many days to create scenarios for")

parent_parser.add_argument('--out-dir', '-o', type=str,
                           default=os.getcwd(), dest='out_dir',
                           help="where generated scenarios will be stored")

parent_parser.add_argument('--scenario-count', '-n', type=int,
                           default=1000, dest='scenario_count',
                           help="how many scenarios to generate")
parent_parser.add_argument('--verbose', '-v', action='count', default=0)

parent_parser.add_argument('--csv', action='store_true',
                           help="store output in .csv format instead of .p.gz")

parent_parser.add_argument('--test', action='store_true')
test_path = Path(Path(__file__).parent.parent, 'test', 'resources')


def run_load():
    args = argparse.ArgumentParser(
        'pgscen-load', parents=[parent_parser],
        description="Create day ahead load scenarios."
        ).parse_args()

    t7k_runner(args.start, args.days, args.out_dir, args.scenario_count,
               create_load=True,
               write_csv=args.csv, verbosity=args.verbose, test=args.test)


def run_wind():
    args = argparse.ArgumentParser(
        'pgscen-wind', parents=[parent_parser],
        description="Create day ahead wind scenarios."
        ).parse_args()

    t7k_runner(args.start, args.days, args.out_dir, args.scenario_count,
               create_wind=True,
               write_csv=args.csv, verbosity=args.verbose, test=args.test)


def run_solar():
    args = argparse.ArgumentParser(
        'pgscen-solar', parents=[parent_parser],
        description="Create day ahead solar scenarios."
        ).parse_args()

    t7k_runner(args.start, args.days, args.out_dir, args.scenario_count,
               create_solar=True,
               write_csv=args.csv, verbosity=args.verbose, test=args.test)


def run_load_solar_joint():
    args = argparse.ArgumentParser(
        'pgscen-load-solar', parents=[parent_parser],
        description="Create day ahead load-solar jointly modeled scenarios."
        ).parse_args()

    t7k_runner(args.start, args.days, args.out_dir, args.scenario_count,
               create_load_solar=True,
               write_csv=args.csv, verbosity=args.verbose, test=args.test)


def run_t7k():
    parser = argparse.ArgumentParser(
        'pgscen', parents=[parent_parser],
        description="Create day-ahead t7k load, wind, and solar scenarios."
        )

    parser.add_argument('--joint', action='store_true',
                        help="use a joint load-solar model")
    args = parser.parse_args()

    t7k_runner(args.start, args.days, args.out_dir, args.scenario_count,
               create_load=not args.joint, create_wind=True,
               create_solar=not args.joint, create_load_solar=args.joint,
               write_csv=args.csv, verbosity=args.verbose, test=args.test)


def t7k_runner(start_date, ndays, out_dir, scen_count,
               create_load=False, create_wind=False,
               create_solar=False, create_load_solar=False,
               write_csv=True, verbosity=0, test=False):
    start = ' '.join([start_date, "06:00:00"])

    if create_load or create_load_solar:
        if test:
            with bz2.BZ2File(Path(test_path, "load.p.gz"), 'r') as f:
                load_zone_actual_df, load_zone_forecast_df = pickle.load(f)
        else:
            load_zone_actual_df, load_zone_forecast_df = load_load_data()

    if create_wind:
        if test:
            with bz2.BZ2File(Path(test_path, "wind.p.gz"), 'r') as f:
                (wind_site_actual_df, wind_site_forecast_df,
                    wind_meta_df) = pickle.load(f)
        else:
            (wind_site_actual_df, wind_site_forecast_df,
                wind_meta_df) = load_wind_data()

    if create_solar or create_load_solar:
        if test:
            with bz2.BZ2File(Path(test_path, "solar.p.gz"), 'r') as f:
                (solar_site_actual_df, solar_site_forecast_df,
                    solar_meta_df) = pickle.load(f)
        else:
            (solar_site_actual_df, solar_site_forecast_df,
                solar_meta_df) = load_solar_data()

    if verbosity >= 2:
        t0 = time.time()

    for scenario_start_time in pd.date_range(start=start, periods=ndays,
                                             freq='D', tz='utc'):
        start_date = scenario_start_time.strftime('%Y-%m-%d')

        scen_timesteps = pd.date_range(start=scenario_start_time,
                                       periods=24, freq='H')

        # TODO: smarter way to account for the end of the year
        if start_date == '2018-12-31':
            break

        out_fl = Path(out_dir, "scens_{}.p.gz".format(start_date))
        if out_fl.exists():
            continue

        if verbosity >= 1:
            print("Creating t7k scenarios for: {}".format(
                scenario_start_time.date()))

        if not write_csv:
            out_scens = dict()

        if create_load or create_load_solar:
            (load_zone_actual_hists,
                load_zone_actual_futures) = split_actuals_hist_future(
                    load_zone_actual_df, scen_timesteps)

            (load_zone_forecast_hists,
                load_zone_forecast_futures) = split_forecasts_hist_future(
                    load_zone_forecast_df, scen_timesteps)

        if create_wind:
            (wind_site_actual_hists,
                wind_site_actual_futures) = split_actuals_hist_future(
                    wind_site_actual_df, scen_timesteps, in_sample=True)
            (wind_site_forecast_hists,
                wind_site_forecast_futures) = split_forecasts_hist_future(
                    wind_site_forecast_df, scen_timesteps, in_sample=True)

        if create_solar or create_load_solar:
            (solar_site_actual_hists,
                solar_site_actual_futures) = split_actuals_hist_future(
                    solar_site_actual_df, scen_timesteps)
            (solar_site_forecast_hists,
                solar_site_forecast_futures) = split_forecasts_hist_future(
                    solar_site_forecast_df, scen_timesteps)

            solar_engn = SolarGeminiEngine(solar_site_actual_hists,
                                           solar_site_forecast_hists,
                                           scenario_start_time, solar_meta_df,
                                           us_state='Texas')

        if create_load:
            load_engn = GeminiEngine(load_zone_actual_hists,
                                     load_zone_forecast_hists,
                                     scenario_start_time, asset_type='load')

            load_engn.fit(5e-2, 5e-2)
            load_engn.create_scenario(scen_count, load_zone_forecast_futures,
                                      bin_width_ratio=0.1, min_sample_size=400)

            if write_csv:
                load_engn.write_to_csv(out_dir, load_zone_actual_futures,
                                       write_forecasts=True)
            else:
                out_scens['Load'] = load_engn.scenarios['load'].round(4)

        if create_wind:
            wind_engn = GeminiEngine(wind_site_actual_hists,
                                     wind_site_forecast_hists,
                                     scenario_start_time, wind_meta_df, 'wind')

            dist = wind_engn.asset_distance().values
            wind_engn.fit(dist / (10 * dist.max()), 5e-2)
            wind_engn.create_scenario(scen_count, wind_site_forecast_futures)

            if write_csv:
                wind_engn.write_to_csv(out_dir, wind_site_actual_futures,
                                       write_forecasts=True)
            else:
                out_scens['Wind'] = wind_engn.scenarios['wind'].round(4)

        if create_solar:
            solar_engn.fit_solar_model(hist_start='2017-01-01',
                                       hist_end='2018-12-31')
            solar_engn.create_solar_scenario(scen_count,
                                             solar_site_forecast_futures)

            if write_csv:
                solar_engn.write_to_csv(out_dir,
                                        {'solar': solar_site_actual_futures},
                                        write_forecasts=True)
            else:
                out_scens['Solar'] = solar_engn.scenarios['solar'].round(4)

        if create_load_solar:
            solar_engn.fit_load_solar_joint_model(
                load_zone_actual_hists, load_zone_forecast_hists,
                hist_start='2020-01-01', hist_end='2020-12-31'
                )

            solar_engn.create_load_solar_joint_scenario(
                scen_count,
                load_zone_forecast_futures, solar_site_forecast_futures
                )

            if write_csv:
                solar_engn.write_to_csv(out_dir,
                                        {'load': load_zone_actual_futures,
                                         'solar': solar_site_actual_futures},
                                        write_forecasts=True)

            else:
                out_scens['Load'] = solar_engn.scenarios['load'].round(4)
                out_scens['Solar'] = solar_engn.scenarios['solar'].round(4)

        if not write_csv:
            with bz2.BZ2File(out_fl, 'w') as f:
                pickle.dump(out_scens, f, protocol=-1)

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
