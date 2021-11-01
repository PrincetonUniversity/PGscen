
import argparse
import os
import pandas as pd
import time

from .utils.data_utils import (load_load_data, load_wind_data, load_solar_data,
                               split_actuals_hist_future,
                               split_forecasts_hist_future,
                               split_actuals_hist_future_wind,
                               split_forecasts_hist_future_wind)
from .engine import GeminiEngine, SolarGeminiEngine


# TODO: make one entry function that handles all possible scenario types?
#       i.e. pgscen --load-solar --wind 2017-10-01 2 /scratch/data ...

parent_parser = argparse.ArgumentParser(add_help=False)

parent_parser.add_argument(
    'start', type=str,
    help="start date for the scenarios, given in YYYY-MM-DD format"
    )
parent_parser.add_argument(
    'days', type=int, help="for how many days to create scenarios for")

parent_parser.add_argument('in_dir', type=str,
                           help="where input datasets are stored")
parent_parser.add_argument('--out-dir', '-o', type=str,
                           default=os.getcwd(), dest='out_dir',
                           help="where generated scenarios will be stored")

parent_parser.add_argument('--scenario-count', '-n', type=int,
                           default=1000, dest='scenario_count',
                           help="how many scenarios to generate")
parent_parser.add_argument('--verbose', '-v', action='count', default=0)


def run_load():
    parser = argparse.ArgumentParser(
        'pgscen-load', parents=[parent_parser],
        description="Create day ahead load scenarios."
        )
    args = parser.parse_args()

    start = ' '.join([args.start, "06:00:00"])
    load_zone_actual_df, load_zone_forecast_df = load_load_data(args.in_dir)

    if args.verbose >= 2:
        t0 = time.time()

    for scenario_start_time in pd.date_range(start=start, periods=args.days,
                                             freq='D', tz='utc'):
        if args.verbose >= 1:
            print("Creating load scenarios for: {}".format(
                scenario_start_time.date()))

        (load_zone_actual_hists,
            load_zone_actual_futures) = split_actuals_hist_future(
                load_zone_actual_df, scenario_start_time)
        (load_zone_forecast_hists,
            load_zone_forecast_futures) = split_forecasts_hist_future(
                load_zone_forecast_df, scenario_start_time)

        ge = GeminiEngine(load_zone_actual_hists, load_zone_forecast_hists,
                          scenario_start_time, asset_type='load')

        ge.fit(5e-2, 5e-2)
        ge.create_scenario(args.scenario_count, load_zone_forecast_futures,
                           bin_width_ratio=0.1, min_sample_size=400)
        ge.write_to_csv(args.out_dir, load_zone_actual_futures,
                        write_forecasts=True)

    if args.verbose >= 2:
        if args.days == 1:
            day_str = "day"
        else:
            day_str = "days"

        print("Created {} load scenarios for {} {} in {:.1f} seconds".format(
            args.scenario_count, args.days, day_str, time.time() - t0))


def run_wind():
    parser = argparse.ArgumentParser(
        'pgscen-wind', parents=[parent_parser],
        description="Create day ahead wind scenarios."
        )
    args = parser.parse_args()

    start = ' '.join([args.start, "06:00:00"])
    (wind_site_actual_df, wind_site_forecast_df,
        wind_meta_df) = load_wind_data(args.in_dir)

    if args.verbose >= 2:
        t0 = time.time()

    for scenario_start_time in pd.date_range(start=start, periods=args.days,
                                             freq='D', tz='utc'):
        if args.verbose >= 1:
            print("Creating wind scenarios for: {}".format(
                scenario_start_time.date()))

        scen_timesteps = pd.date_range(start=scenario_start_time,
                                       periods=24, freq='H')

        (wind_site_actual_hists,
            wind_site_actual_futures) = split_actuals_hist_future_wind(
                    wind_site_actual_df, scen_timesteps)
        (wind_site_forecast_hists,
            wind_site_forecast_futures) = split_forecasts_hist_future_wind(
                    wind_site_forecast_df, scen_timesteps)

        ge = GeminiEngine(wind_site_actual_hists, wind_site_forecast_hists,
                          scenario_start_time, wind_meta_df, 'wind')

        dist = ge.asset_distance().values
        ge.fit(dist / (10 * dist.max()), 5e-2)

        ge.create_scenario(args.scenario_count, wind_site_forecast_futures)
        ge.write_to_csv(args.out_dir, wind_site_actual_futures,
                        write_forecasts=True)

    if args.verbose >= 2:
        if args.days == 1:
            day_str = "day"
        else:
            day_str = "days"

        print("Created {} wind scenarios for {} {} in {:.1f} seconds".format(
            args.scenario_count, args.days, day_str, time.time() - t0))


def run_solar():
    parser = argparse.ArgumentParser(
        'pgscen-solar', parents=[parent_parser],
        description="Create day ahead solar scenarios."
        )
    args = parser.parse_args()

    start = ' '.join([args.start, "06:00:00"])
    (solar_site_actual_df, solar_site_forecast_df,
        solar_meta_df) = load_solar_data(args.in_dir)

    if args.verbose >= 2:
        t0 = time.time()

    for scenario_start_time in pd.date_range(start=start, periods=args.days,
                                             freq='D', tz='utc'):
        if args.verbose >= 1:
            print("Creating solar scenarios for: {}".format(
                scenario_start_time.date()))

        (solar_site_actual_hists,
            solar_site_actual_futures) = split_actuals_hist_future(
                    solar_site_actual_df, scenario_start_time)
        (solar_site_forecast_hists,
            solar_site_forecast_futures) = split_forecasts_hist_future(
                    solar_site_forecast_df, scenario_start_time)

        se = SolarGeminiEngine(solar_site_actual_hists,
                               solar_site_forecast_hists,
                               scenario_start_time, solar_meta_df)

        se.fit_solar_model()
        se.create_solar_scenario(args.scenario_count,
                                 solar_site_forecast_futures)

        se.write_to_csv(args.out_dir, {'solar': solar_site_actual_futures},
                        write_forecasts=True)

    if args.verbose >= 2:
        if args.days == 1:
            day_str = "day"
        else:
            day_str = "days"

        print("Created {} solar scenarios for {} {} in {:.1f} seconds".format(
            args.scenario_count, args.days, day_str, time.time() - t0))


def run_load_solar_joint():
    parser = argparse.ArgumentParser(
        'pgscen-load-solar', parents=[parent_parser],
        description="Create day ahead load-solar jointly modeled scenarios."
        )
    args = parser.parse_args()

    start = ' '.join([args.start, "06:00:00"])
    load_zone_actual_df, load_zone_forecast_df = load_load_data(args.in_dir)
    (solar_site_actual_df, solar_site_forecast_df,
        solar_meta_df) = load_solar_data(args.in_dir)

    if args.verbose >= 2:
        t0 = time.time()

    for scenario_start_time in pd.date_range(start=start, periods=args.days,
                                             freq='D', tz='utc'):
        if args.verbose >= 1:
            print("Creating load-solar joint scenarios for: {}".format(
                scenario_start_time.date()))

        (solar_site_actual_hists,
            solar_site_actual_futures) = split_actuals_hist_future(
                    solar_site_actual_df, scenario_start_time)
        (solar_site_forecast_hists,
            solar_site_forecast_futures) = split_forecasts_hist_future(
                    solar_site_forecast_df, scenario_start_time)

        (load_zone_actual_hists,
            load_zone_actual_futures) = split_actuals_hist_future(
                    load_zone_actual_df, scenario_start_time)
        (load_zone_forecast_hists,
            load_zone_forecast_futures) = split_forecasts_hist_future(
                    load_zone_forecast_df, scenario_start_time)

        se = SolarGeminiEngine(solar_site_actual_hists,
                               solar_site_forecast_hists,
                               scenario_start_time, solar_meta_df)

        se.fit_load_solar_joint_model(load_zone_actual_hists,
                                      load_zone_forecast_hists)
        se.create_load_solar_joint_scenario(args.scenario_count,
                                            load_zone_forecast_futures,
                                            solar_site_forecast_futures)

        se.write_to_csv(args.out_dir, {'load': load_zone_actual_futures,
                                       'solar': solar_site_actual_futures},
                        write_forecasts=True)

    if args.verbose >= 2:
        if args.days == 1:
            day_str = "day"
        else:
            day_str = "days"

        print("Created {} load-solar joint scenarios for {} {} in "
              "{:.1f} seconds".format(args.scenario_count,
                                      args.days, day_str, time.time() - t0))
