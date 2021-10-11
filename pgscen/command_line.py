
import argparse
import os
import pandas as pd
from pathlib import Path
import time
from pgscen.iso.ercot import create_day_ahead_load_solar_joint_scenario


def run_load_solar_joint():
    parser = argparse.ArgumentParser(
        'pgscen-load-solar',
        description="Create day ahead load-solar jointly modeled scenarios."
        )

    parser.add_argument(
        'start', type=str,
        help="start date for the scenarios, given in YYYY-MM-DD format"
        )

    parser.add_argument('days', type=int,
                        help="for how many days to create scenarios for")

    parser.add_argument('--out-dir', '-o', type=str,
                        default=os.getcwd(), dest='out_dir',
                        help="where generated scenarios will be stored")

    parser.add_argument('--scenario-count', '-n', type=int,
                        default=1000, dest='scenario_count',
                        help="how many scenarios to generate")

    parser.add_argument('--verbose', '-v', action='count', default=0)
    args = parser.parse_args()
    input_dir = Path(Path(__file__).parent.resolve(), '..', 'data')

    load_zone_actual_df = pd.read_csv(
        Path(input_dir, 'Load', 'ERCOT', 'Actual',
             'load_actual_1h_zone_2017_2018_utc.csv'),
        parse_dates=['Time'], index_col='Time'
        )

    load_zone_forecast_df = pd.read_csv(
        Path(input_dir, 'Load', 'ERCOT', 'Day-ahead',
             'load_day_ahead_forecast_zone_2017_2018_utc.csv'),
        parse_dates=['Issue_time', 'Forecast_time']
        )

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

    load_zone_list = load_zone_actual_df.columns.tolist()
    solar_site_list = solar_site_actual_df.columns.tolist()
    start = ' '.join([args.start, "06:00:00"])

    if args.verbose >= 2:
        t0 = time.time()

    for scenario_start_time in pd.date_range(start=start, periods=args.days,
                                             freq='D', tz='utc'):
        if args.verbose >= 1:
            print("Creating scenarios for: {}".format(scenario_start_time))

        load_zone_hist_actual_df = load_zone_actual_df[
            load_zone_actual_df.index < scenario_start_time]
        load_zone_hist_forecast_df = load_zone_forecast_df[
            load_zone_forecast_df['Forecast_time'] < scenario_start_time]

        load_zone_future_actual_df = load_zone_actual_df[
            load_zone_actual_df.index >= scenario_start_time]
        load_zone_future_forecast_df = load_zone_forecast_df[
            load_zone_forecast_df['Forecast_time'] >= scenario_start_time]

        solar_site_hist_actual_df = solar_site_actual_df[
            solar_site_actual_df.index < scenario_start_time]
        solar_site_hist_forecast_df = solar_site_forecast_df[
            solar_site_forecast_df['Forecast_time'] < scenario_start_time]

        solar_site_future_actual_df = solar_site_actual_df[
            solar_site_actual_df.index >= scenario_start_time]
        solar_site_future_forecast_df = solar_site_forecast_df[
            solar_site_forecast_df['Forecast_time'] >= scenario_start_time]

        se = create_day_ahead_load_solar_joint_scenario(
            args.scenario_count, scenario_start_time, load_zone_list,
            load_zone_hist_forecast_df, load_zone_hist_actual_df,
            solar_meta_df, solar_site_list, solar_site_hist_forecast_df,
            solar_site_hist_actual_df, load_zone_future_forecast_df,
            solar_site_future_forecast_df,
            load_future_actual_df=load_zone_future_actual_df,
            solar_future_actual_df=solar_site_future_actual_df,
            output_dir=args.out_dir, return_engine=True
            )

    if args.verbose >= 2:
        if args.days == 1:
            day_str = "day"
        else:
            day_str = "days"

        print("Created {} scenarios for {} {} in {:.1f} seconds".format(
            args.scenario_count, args.days, day_str, time.time() - t0))
