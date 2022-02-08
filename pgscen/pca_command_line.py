"""Command line interface for generating scenarios with ERCOT/NREL datasets."""

import argparse
import os
import pandas as pd
from pathlib import Path
import bz2
import dill as pickle
import time

from .utils.data_utils import (load_solar_data,
                               split_actuals_hist_future,
                               split_forecasts_hist_future)
from .pca import PCAGeminiEngine


# TODO: make one entry function that handles all possible scenario types?
#       i.e. pgscen --load-solar --wind 2017-10-01 2 /scratch/data ...

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

parent_parser.add_argument('--test', action='store_true')
test_path = Path(Path(__file__).parent.parent, 'test', 'resources')


def run_solar():
    parser = argparse.ArgumentParser(
        'pgscen-pca-solar', parents=[parent_parser],
        description="Create day ahead solar scenarios."
        )

    args = parser.parse_args()
    start = ' '.join([args.start, "06:00:00"])

    if args.test:
        with bz2.BZ2File(Path(test_path, "solar.p.gz"), 'r') as f:
            (solar_site_actual_df, solar_site_forecast_df,
                solar_meta_df) = pickle.load(f)

    else:
        (solar_site_actual_df, solar_site_forecast_df,
            solar_meta_df) = load_solar_data()

    if args.verbose >= 2:
        t0 = time.time()

    for scenario_start_time in pd.date_range(start=start, periods=args.days,
                                             freq='D', tz='utc'):
        if args.verbose >= 1:
            print("Creating solar scenarios for: {}".format(
                scenario_start_time.date()))

        scen_timesteps = pd.date_range(start=scenario_start_time,
                                       periods=24, freq='H')

        (solar_site_actual_hists,
            solar_site_actual_futures) = split_actuals_hist_future(
                    solar_site_actual_df, scen_timesteps)
        (solar_site_forecast_hists,
            solar_site_forecast_futures) = split_forecasts_hist_future(
                    solar_site_forecast_df, scen_timesteps)

        se = PCAGeminiEngine(solar_site_actual_hists,
                               solar_site_forecast_hists,
                               scenario_start_time, solar_meta_df)

        dist = se.asset_distance().values
        se.fit(10, dist / (10 * dist.max()), 5e-2)
        se.create_scenario(args.scenario_count, solar_site_forecast_futures)
        se.write_to_csv(args.out_dir, {'solar': solar_site_actual_futures},
                        write_forecasts=True)

    if args.verbose >= 2:
        if args.days == 1:
            day_str = "day"
        else:
            day_str = "days"

        print("Created {} solar scenarios for {} {} in {:.1f} seconds".format(
            args.scenario_count, args.days, day_str, time.time() - t0))
