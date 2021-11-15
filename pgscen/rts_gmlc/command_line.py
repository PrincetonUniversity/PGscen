
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


def run_rts():
    parser = argparse.ArgumentParser(
        'pgscen-rts', parents=[parent_parser],
        description="Create day ahead load-solar jointly modeled scenarios."
        )

    parser.add_argument('rts_dir', type=str,
                        help="where RTS-GMLC repository is stored")
    parser.add_argument('--csv', action='store_true',
                        help="store output in .csv format instead of .p.gz")

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
        start_date = scenario_start_time.strftime('%Y-%m-%d')

        scen_timesteps = pd.date_range(start=scenario_start_time,
                                       periods=24, freq='H')

        # TODO: smarter way to account for the end of the year
        if start_date == '2020-12-31':
            break

        out_fl = Path(args.out_dir, "scens_{}.p.gz".format(start_date))
        if out_fl.exists():
            continue

        if args.verbose >= 1:
            print("Creating RTS-GMLC scenarios for: {}".format(
                scenario_start_time.date()))

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

        if args.csv:
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

        if args.csv:
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
