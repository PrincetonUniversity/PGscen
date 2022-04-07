"""Interface for generating scenarios using PCA models on Texas-7k datasets."""

import argparse
import pandas as pd
from pathlib import Path
import bz2
import dill as pickle
import time

from .command_line import parent_parser
from .utils.data_utils import (load_solar_data, load_load_data,
                               split_actuals_hist_future,
                               split_forecasts_hist_future)
from .pca import PCAGeminiEngine


pca_parser = argparse.ArgumentParser(add_help=False, parents=[parent_parser])
pca_parser.add_argument('--components', '-c', type=int, default=12,
                        help="how many factors to use when fitting the PCA")


def run_solar():
    args = argparse.ArgumentParser(
        'pgscen-pca-solar', parents=[pca_parser],
        description="Create day ahead solar scenarios using PCA features."
        ).parse_args()

    t7k_pca_runner(args.start, args.days, args.out_dir, args.scenario_count,
                   args.components, create_load_solar=False,
                   write_csv=not args.pickle, skip_existing=args.skip_existing,
                   verbosity=args.verbose)


def run_load_solar():
    args = argparse.ArgumentParser(
        'pgscen-pca-load-solar', parents=[pca_parser],
        description="Create day ahead load-solar jointly modeled scenarios "
                    "using PCA features."
        ).parse_args()

    t7k_pca_runner(args.start, args.days, args.out_dir, args.scenario_count,
                   args.components, create_load_solar=True,
                   write_csv=not args.pickle, skip_existing=args.skip_existing,
                   verbosity=args.verbose)


def t7k_pca_runner(start_date, ndays, out_dir, scen_count, components,
                   create_load_solar=False,
                   write_csv=True, skip_existing=False, verbosity=0):
    start = ' '.join([start_date, "06:00:00"])

    # load input datasets, starting with solar farm data
    (solar_site_actual_df, solar_site_forecast_df,
        solar_meta_df) = load_solar_data()

    if create_load_solar:
        load_zone_actual_df, load_zone_forecast_df = load_load_data()

    if verbosity >= 2:
        t0 = time.time()

    # create scenarios for each requested day
    for scenario_start_time in pd.date_range(start=start, periods=ndays,
                                             freq='D', tz='utc'):
        date_lbl = scenario_start_time.strftime('%Y-%m-%d')

        scen_timesteps = pd.date_range(start=scenario_start_time,
                                       periods=24, freq='H')

        if verbosity >= 1:
            print("Creating t7k scenarios for: {}".format(date_lbl))

        # don't generate scenarios for this day if they have already been saved
        # in this output directory
        if not write_csv:
            out_fl = Path(out_dir, "scens_{}.p.gz".format(date_lbl))

            if skip_existing and out_fl.exists():
                continue

            out_scens = dict()

        # split input datasets into training and testing subsets
        (solar_site_actual_hists,
            solar_site_actual_futures) = split_actuals_hist_future(
                solar_site_actual_df, scen_timesteps)
        (solar_site_forecast_hists,
            solar_site_forecast_futures) = split_forecasts_hist_future(
                solar_site_forecast_df, scen_timesteps)

        if create_load_solar:
            (load_zone_actual_hists,
                load_zone_actual_futures) = split_actuals_hist_future(
                    load_zone_actual_df, scen_timesteps)

            (load_zone_forecast_hists,
                load_zone_forecast_futures) = split_forecasts_hist_future(
                    load_zone_forecast_df, scen_timesteps)

        solar_engn = PCAGeminiEngine(solar_site_actual_hists,
                                     solar_site_forecast_hists,
                                     scenario_start_time, solar_meta_df)
        dist = solar_engn.asset_distance().values

        if create_load_solar:
            solar_engn.fit_load_solar_joint_model(
                load_hist_actual_df=load_zone_actual_hists,
                load_hist_forecast_df=load_zone_forecast_hists,
                asset_rho=dist / (10 * dist.max()), horizon_rho=5e-2,
                num_of_components=components
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

        else:
            solar_engn.fit(asset_rho=dist / (10 * dist.max()),
                           horizon_rho=5e-2, num_of_components=components)
            solar_engn.create_scenario(scen_count, solar_site_forecast_futures)

            if write_csv:
                solar_engn.write_to_csv(out_dir,
                                        {'solar': solar_site_actual_futures},
                                        write_forecasts=True)

            else:
                out_scens['Solar'] = solar_engn.scenarios['solar'].round(4)

        if not write_csv:
            with bz2.BZ2File(out_fl, 'w') as f:
                pickle.dump(out_scens, f, protocol=-1)

    if verbosity >= 2:
        if ndays == 1:
            day_str = "day"
        else:
            day_str = "days"

        if create_load_solar:
            type_str = 'joint load-solar'
        else:
            type_str = 'solar'

        print("Created {} {} scenarios for {} {} in {:.1f} seconds".format(
            scen_count, type_str, ndays, day_str, time.time() - t0))
