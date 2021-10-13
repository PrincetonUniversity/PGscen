
import argparse
import os
import pandas as pd
from pathlib import Path
import time

from .engine import GeminiEngine, SolarGeminiEngine


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


# TODO: make one entry function that handles all possible scenario types?
#       i.e. pgscen --load-solar --wind 2017-10-01 2 /scratch/data ...

def load_solar_data(input_dir):
    solar_site_actual_df = pd.read_csv(
        Path(input_dir, 'Solar', 'NREL', 'Actual',
             'solar_actual_1h_site_2017_2018_utc.csv'),
        parse_dates=['Time'], index_col='Time'
        )

    solar_site_forecast_df = pd.read_csv(
        Path(input_dir, 'Solar', 'NREL', 'Day-ahead',
             'solar_day_ahead_forecast_site_2017_2018_utc.csv'),
        parse_dates=['Issue_time', 'Forecast_time']
        )

    solar_meta_df = pd.read_excel(
        Path(input_dir, 'MetaData', 'solar_meta.xlsx'))

    return solar_site_actual_df, solar_site_forecast_df, solar_meta_df


def split_actuals_hist_future(actual_df, scen_start_time):
    hist_df = actual_df[actual_df.index < scen_start_time]
    future_df = actual_df[actual_df.index >= scen_start_time]

    return hist_df, future_df


def split_forecasts_hist_future(forecast_df, scen_start_time):
    hist_df = forecast_df[forecast_df.Forecast_time < scen_start_time]
    future_df = forecast_df[forecast_df.Forecast_time >= scen_start_time]

    return hist_df, future_df


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
            print("Creating scenarios for: {}".format(scenario_start_time))

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

        print("Created {} scenarios for {} {} in {:.1f} seconds".format(
            args.scenario_count, args.days, day_str, time.time() - t0))


def run_load_solar_joint():
    parser = argparse.ArgumentParser(
        'pgscen-load-solar', parents=[parent_parser],
        description="Create day ahead load-solar jointly modeled scenarios."
        )
    args = parser.parse_args()

    load_zone_actual_df = pd.read_csv(
        Path(args.in_dir, 'Load', 'ERCOT', 'Actual',
             'load_actual_1h_zone_2017_2018_utc.csv'),
        parse_dates=['Time'], index_col='Time'
        )

    load_zone_forecast_df = pd.read_csv(
        Path(args.in_dir, 'Load', 'ERCOT', 'Day-ahead',
             'load_day_ahead_forecast_zone_2017_2018_utc.csv'),
        parse_dates=['Issue_time', 'Forecast_time']
        )

    start = ' '.join([args.start, "06:00:00"])
    (solar_site_actual_df, solar_site_forecast_df,
        solar_meta_df) = load_solar_data(args.in_dir)

    if args.verbose >= 2:
        t0 = time.time()

    for scenario_start_time in pd.date_range(start=start, periods=args.days,
                                             freq='D', tz='utc'):
        if args.verbose >= 1:
            print("Creating scenarios for: {}".format(scenario_start_time))

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

        print("Created {} scenarios for {} {} in {:.1f} seconds".format(
            args.scenario_count, args.days, day_str, time.time() - t0))