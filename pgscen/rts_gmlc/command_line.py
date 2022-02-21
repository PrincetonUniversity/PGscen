"""Command line interface for generating scenarios for the RTS-GMLC system."""

import argparse
from pathlib import Path
import pandas as pd
import time
import bz2
import dill as pickle

from ..command_line import parent_parser
from ..engine import GeminiEngine, SolarGeminiEngine

from .data_utils import load_load_data, load_wind_data, load_solar_data
from ..utils.data_utils import (split_actuals_hist_future,
                                split_forecasts_hist_future)


rts_parser = argparse.ArgumentParser(add_help=False, parents=[parent_parser])
rts_parser.add_argument('rts_dir', type=str,
                        help="where RTS-GMLC repository is stored")


#TODO: consolidate the code for these interfaces like we do for t7k?
def run_rts():
    parser = argparse.ArgumentParser(
        'pgscen-rts', parents=[rts_parser],
        description="Create day-ahead RTS load, wind, and solar scenarios."
        )

    args = parser.parse_args()
    start = ' '.join([args.start, "08:00:00"])

    load_zone_actual_df, load_zone_forecast_df = load_load_data(args.rts_dir)
    (wind_site_actual_df, wind_site_forecast_df,
        wind_meta_df) = load_wind_data(args.rts_dir)
    (solar_site_actual_df, solar_site_forecast_df,
        solar_meta_df) = load_solar_data(args.rts_dir)

    if args.verbose >= 2:
        t0 = time.time()

    for scenario_start_time in pd.date_range(start=start, periods=args.days,
                                             freq='D', tz='utc'):
        date_lbl = scenario_start_time.strftime('%Y-%m-%d')

        scen_timesteps = pd.date_range(start=scenario_start_time,
                                       periods=24, freq='H')

        if args.verbose >= 1:
            print("Creating RTS-GMLC load+wind+solar scenarios for: {}".format(
                date_lbl))

        if args.pickle:
            out_fl = Path(args.out_dir, "scens_{}.p.gz".format(date_lbl))

            if args.skip_existing and out_fl.exists():
                continue

        # for RTS we always do in-sample since we only have a year of data
        (load_zone_actual_hists,
            load_zone_actual_futures) = split_actuals_hist_future(
                    load_zone_actual_df, scen_timesteps, in_sample=True)
        (load_zone_forecast_hists,
            load_zone_forecast_futures) = split_forecasts_hist_future(
                    load_zone_forecast_df, scen_timesteps, in_sample=True)

        (wind_site_actual_hists,
            wind_site_actual_futures) = split_actuals_hist_future(
                    wind_site_actual_df, scen_timesteps, in_sample=True)
        (wind_site_forecast_hists,
            wind_site_forecast_futures) = split_forecasts_hist_future(
                    wind_site_forecast_df, scen_timesteps, in_sample=True)

        (solar_site_actual_hists,
            solar_site_actual_futures) = split_actuals_hist_future(
                    solar_site_actual_df, scen_timesteps, in_sample=True)
        (solar_site_forecast_hists,
            solar_site_forecast_futures) = split_forecasts_hist_future(
                    solar_site_forecast_df, scen_timesteps, in_sample=True)

        load_engn = GeminiEngine(load_zone_actual_hists,
                                 load_zone_forecast_hists,
                                 scenario_start_time, asset_type='load')

        load_engn.fit(5e-2, 5e-2)
        load_engn.create_scenario(args.scenario_count,
                                  load_zone_forecast_futures,
                                  bin_width_ratio=0.1, min_sample_size=400)

        if not args.pickle:
            load_engn.write_to_csv(args.out_dir, load_zone_actual_futures,
                                   write_forecasts=True)

        wind_engn = GeminiEngine(wind_site_actual_hists,
                                 wind_site_forecast_hists,
                                 scenario_start_time, wind_meta_df, 'wind')

        dist = wind_engn.asset_distance().values
        wind_engn.fit(dist / (10 * dist.max()), 5e-2)
        wind_engn.create_scenario(args.scenario_count,
                                  wind_site_forecast_futures)

        if not args.pickle:
            wind_engn.write_to_csv(args.out_dir, wind_site_actual_futures,
                                   write_forecasts=True)

        solar_engn = SolarGeminiEngine(solar_site_actual_hists,
                                       solar_site_forecast_hists,
                                       scenario_start_time, solar_meta_df,
                                       us_state='California')

        solar_engn.fit_solar_model(
            hist_start='2020-01-01', hist_end='2020-12-31')
        solar_engn.create_solar_scenario(args.scenario_count,
                                         solar_site_forecast_futures)

        if not args.pickle:
            solar_engn.write_to_csv(args.out_dir,
                                    {'solar': solar_site_actual_futures},
                                    write_forecasts=True)

        else:
            with bz2.BZ2File(out_fl, 'w') as f:
                pickle.dump({'Load': load_engn.scenarios['load'].round(4),
                             'Wind': wind_engn.scenarios['wind'].round(4),
                             'Solar': solar_engn.scenarios['solar'].round(4)},
                            f, protocol=-1)

    if args.verbose >= 2:
        if args.days == 1:
            day_str = "day"
        else:
            day_str = "days"

        print("Created {} each of load, wind, and solar scenarios for {} {} "
              "in {:.1f} seconds".format(args.scenario_count,
                                         args.days, day_str, time.time() - t0))


def run_rts_joint():
    parser = argparse.ArgumentParser(
        'pgscen-rts-joint', parents=[rts_parser],
        description="Create day-ahead RTS wind and load-solar joint scenarios."
        )

    args = parser.parse_args()
    start = ' '.join([args.start, "08:00:00"])

    load_zone_actual_df, load_zone_forecast_df = load_load_data(args.rts_dir)
    (wind_site_actual_df, wind_site_forecast_df,
        wind_meta_df) = load_wind_data(args.rts_dir)
    (solar_site_actual_df, solar_site_forecast_df,
        solar_meta_df) = load_solar_data(args.rts_dir)

    if args.verbose >= 2:
        t0 = time.time()

    for scenario_start_time in pd.date_range(start=start, periods=args.days,
                                             freq='D', tz='utc'):
        date_lbl = scenario_start_time.strftime('%Y-%m-%d')

        scen_timesteps = pd.date_range(start=scenario_start_time,
                                       periods=24, freq='H')

        if args.verbose >= 1:
            print("Creating RTS-GMLC wind+joint scenarios for: {}".format(
                date_lbl))

        if args.pickle:
            out_fl = Path(args.out_dir, "scens_{}.p.gz".format(date_lbl))

            if args.skip_existing and out_fl.exists():
                continue

        (load_zone_actual_hists,
            load_zone_actual_futures) = split_actuals_hist_future(
                    load_zone_actual_df, scen_timesteps, in_sample=True)
        (load_zone_forecast_hists,
            load_zone_forecast_futures) = split_forecasts_hist_future(
                    load_zone_forecast_df, scen_timesteps, in_sample=True)

        (wind_site_actual_hists,
            wind_site_actual_futures) = split_actuals_hist_future(
                    wind_site_actual_df, scen_timesteps, in_sample=True)
        (wind_site_forecast_hists,
            wind_site_forecast_futures) = split_forecasts_hist_future(
                    wind_site_forecast_df, scen_timesteps, in_sample=True)

        (solar_site_actual_hists,
            solar_site_actual_futures) = split_actuals_hist_future(
                    solar_site_actual_df, scen_timesteps, in_sample=True)
        (solar_site_forecast_hists,
            solar_site_forecast_futures) = split_forecasts_hist_future(
                    solar_site_forecast_df, scen_timesteps, in_sample=True)

        ge = GeminiEngine(wind_site_actual_hists, wind_site_forecast_hists,
                          scenario_start_time, wind_meta_df, 'wind')

        dist = ge.asset_distance().values
        ge.fit(dist / (10 * dist.max()), 5e-2)
        ge.create_scenario(args.scenario_count, wind_site_forecast_futures)

        if not args.pickle:
            ge.write_to_csv(args.out_dir, wind_site_actual_futures,
                            write_forecasts=True)

        se = SolarGeminiEngine(solar_site_actual_hists,
                               solar_site_forecast_hists,
                               scenario_start_time, solar_meta_df,
                               us_state='California')

        se.fit_load_solar_joint_model(
            load_zone_actual_hists, load_zone_forecast_hists,
            hist_start='2020-01-01', hist_end='2020-12-31'
            )

        se.create_load_solar_joint_scenario(args.scenario_count,
                                            load_zone_forecast_futures,
                                            solar_site_forecast_futures)

        if not args.pickle:
            se.write_to_csv(args.out_dir, {'load': load_zone_actual_futures,
                                           'solar': solar_site_actual_futures},
                            write_forecasts=True)

        else:
            with bz2.BZ2File(out_fl, 'w') as f:
                pickle.dump({'Wind': ge.scenarios['wind'].round(4),
                             'Load': se.scenarios['load'].round(4),
                             'Solar': se.scenarios['solar'].round(4)},
                            f, protocol=-1)

    if args.verbose >= 2:
        if args.days == 1:
            day_str = "day"
        else:
            day_str = "days"

        print("Created {} wind and load-solar joint scenarios for {} {} in "
              "{:.1f} seconds".format(args.scenario_count,
                                      args.days, day_str, time.time() - t0))
