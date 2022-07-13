"""Interface for generating scenarios using PCA models on RTS-GMLC datasets."""

import argparse
import os
from pathlib import Path
import shutil
import time
import bz2
import dill as pickle
from typing import Union

import numpy as np
import pandas as pd

from ..pca import PCAGeminiEngine
from .command_line import rts_runner
from ..pca_command_line import pca_parser, parse_component
from ..scoring import compute_energy_scores, compute_variograms

from .data_utils import load_load_data, load_solar_data
from ..utils.data_utils import (split_actuals_hist_future,
                                split_forecasts_hist_future)


rts_pca_parser = argparse.ArgumentParser(add_help=False, parents=[pca_parser])
rts_pca_parser.add_argument('rts_dir', type=str,
                            help="where RTS-GMLC repository is stored")

joint_parser = argparse.ArgumentParser(parents=[rts_pca_parser],
                                       add_help=False)
joint_parser.add_argument('--use-all-load-history',
                          action='store_true', dest='use_all_load_hist',
                          help="train load models using all out-of-sample "
                               "historical days instead of the same "
                               "window used for solar models")


def run_solar() -> None:
    args = argparse.ArgumentParser(
        'pgscen-rts-pca-solar', parents=[rts_pca_parser],
        description="Create day ahead RTS solar scenarios using PCA features."
        ).parse_args()

    rts_pca_runner(args.start, args.days, args.rts_dir, args.out_dir,
                   args.scenario_count, args.components, args.nearest_days,
                   args.random_seed, create_load_solar=False,
                   write_csv=not args.pickle, skip_existing=args.skip_existing,
                   get_energy_scores=args.energy_scores,
                   get_variograms=args.variograms,
                   verbosity=args.verbose)


def run_load_solar() -> None:
    args = argparse.ArgumentParser(
        'pgscen-rts-pca-load-solar', parents=[joint_parser],
        description="Create day ahead RTS load-solar joint-modeled scenarios."
        ).parse_args()

    rts_pca_runner(args.start, args.days, args.rts_dir, args.out_dir,
                   args.scenario_count, args.components, args.nearest_days,
                   args.random_seed, create_load_solar=True,
                   write_csv=not args.pickle, skip_existing=args.skip_existing,
                   get_energy_scores=args.energy_scores,
                   get_variograms=args.variograms,
                   use_all_load_hist=args.use_all_load_hist,
                   verbosity=args.verbose)


def run_rts_pca() -> None:
    parser = argparse.ArgumentParser(
        'pgscen-rts-pca', parents=[joint_parser],
        description="Create day-ahead RTS load, wind, and solar PCA scenarios."
        )

    parser.add_argument('--joint', action='store_true',
                        help="use a joint load-solar model")
    args = parser.parse_args()

    rts_runner(args.start, args.days, args.rts_dir, args.out_dir,
               args.scenario_count, args.nearest_days, args.random_seed,
               create_load=not args.joint, create_wind=True,
               write_csv=not args.pickle, skip_existing=args.skip_existing,
               get_energy_scores=args.energy_scores,
               get_variograms=args.variograms,
               verbosity=args.verbose)

    # to avoid output files from the non-PCA scenarios from being overwritten,
    # we move them to a temporary directory before creating PCA scenarios
    if args.pickle:
        os.makedirs(str(Path(args.out_dir, 'tmp')), exist_ok=True)

        scen_lbls = ["scens_{}.p.gz".format(start_time.strftime('%Y-%m-%d'))
                     for start_time in pd.date_range(start=args.start,
                                                     periods=args.days,
                                                     freq='D', tz='utc')]

        for scen_lbl in scen_lbls:
            os.rename(str(Path(args.out_dir, scen_lbl)),
                      str(Path(args.out_dir, 'tmp', scen_lbl)))

    if args.energy_scores:
        os.makedirs(str(Path(args.out_dir, 'tmp')), exist_ok=True)

        scen_lbls = ["escores_{}.p.gz".format(start_time.strftime('%Y-%m-%d'))
                     for start_time in pd.date_range(start=args.start,
                                                     periods=args.days,
                                                     freq='D', tz='utc')]

        for scen_lbl in scen_lbls:
            os.rename(str(Path(args.out_dir, scen_lbl)),
                      str(Path(args.out_dir, 'tmp', scen_lbl)))

    if args.variograms:
        os.makedirs(str(Path(args.out_dir, 'tmp')), exist_ok=True)

        scen_lbls = ["varios_{}.p.gz".format(start_time.strftime('%Y-%m-%d'))
                     for start_time in pd.date_range(start=args.start,
                                                     periods=args.days,
                                                     freq='D', tz='utc')]

        for scen_lbl in scen_lbls:
            os.rename(str(Path(args.out_dir, scen_lbl)),
                      str(Path(args.out_dir, 'tmp', scen_lbl)))

    rts_pca_runner(args.start, args.days, args.rts_dir, args.out_dir,
                   args.scenario_count, args.components, args.nearest_days,
                   args.random_seed, create_load_solar=args.joint,
                   write_csv=not args.pickle, skip_existing=args.skip_existing,
                   get_energy_scores=args.energy_scores,
                   get_variograms=args.variograms,
                   use_all_load_hist=args.use_all_load_hist,
                   verbosity=args.verbose)

    # to create the final output files we merge the PCA and the non-PCA outputs
    if args.pickle:
        for scen_lbl in scen_lbls:
            with bz2.BZ2File(Path(args.out_dir, scen_lbl), 'r') as f:
                pca_scens = pickle.load(f)
            with bz2.BZ2File(Path(args.out_dir, 'tmp', scen_lbl), 'r') as f:
                pca_scens.update(pickle.load(f))

            os.remove(Path(args.out_dir, 'tmp', scen_lbl))
            with bz2.BZ2File(Path(args.out_dir, scen_lbl), 'w') as f:
                pickle.dump(pca_scens, f, protocol=-1)


#TODO: lot of overlap here with the t7k PCA runner
def rts_pca_runner(start_date: str, ndays: int, rts_dir: str, out_dir: str,
                   scen_count: int, components: str,
                   nearest_days: Union[int, None], random_seed: int,
                   create_load_solar: bool = False,
                   write_csv: bool = True, skip_existing: bool = False,
                   get_energy_scores: bool = False,
                   get_variograms: bool = False,
                   use_all_load_hist: bool = False,
                   verbosity: int = 0) -> None:
    start = ' '.join([start_date, "08:00:00"])

    if random_seed:
        np.random.seed(random_seed)

    components = parse_component(components)
    if nearest_days is None:
        nearest_days = 50

    # load input datasets, starting with solar farm data
    (solar_site_actual_df, solar_site_forecast_df,
        solar_meta_df) = load_solar_data(rts_dir)

    if create_load_solar:
        load_zone_actual_df, load_zone_forecast_df = load_load_data(rts_dir)

    if verbosity >= 2:
        t0 = time.time()

    # create scenarios for each requested day
    for scenario_start_time in pd.date_range(start=start, periods=ndays,
                                             freq='D', tz='utc'):
        date_lbl = scenario_start_time.strftime('%Y-%m-%d')

        scen_timesteps = pd.date_range(start=scenario_start_time,
                                       periods=24, freq='H')

        if verbosity >= 1:
            print("Creating RTS-GMLC PCA scenarios for: {}".format(date_lbl))

        if get_energy_scores:
            energy_scores = dict()
        if get_variograms:
            variograms = dict()

        # don't generate scenarios for this day if they have already been saved
        # in this output directory
        if not write_csv:
            out_fl = Path(out_dir, "scens_{}.p.gz".format(date_lbl))

            if skip_existing and out_fl.exists():
                continue

            out_scens = dict()

        if write_csv and skip_existing:
            date_path = Path(out_dir, scenario_start_time.strftime('%Y%m%d'))

            if create_load_solar and Path(date_path, 'load').exists():
                shutil.rmtree(Path(date_path, 'load'))

            if Path(date_path, 'solar').exists():
                shutil.rmtree(Path(date_path, 'solar'))

        # split input datasets into training and testing subsets
        (solar_site_actual_hists,
            solar_site_actual_futures) = split_actuals_hist_future(
                solar_site_actual_df, scen_timesteps, in_sample=True)
        (solar_site_forecast_hists,
            solar_site_forecast_futures) = split_forecasts_hist_future(
                solar_site_forecast_df, scen_timesteps, in_sample=True)

        if create_load_solar:
            (load_zone_actual_hists,
                load_zone_actual_futures) = split_actuals_hist_future(
                    load_zone_actual_df, scen_timesteps, in_sample=True)

            (load_zone_forecast_hists,
                load_zone_forecast_futures) = split_forecasts_hist_future(
                    load_zone_forecast_df, scen_timesteps, in_sample=True)

        solar_engn = PCAGeminiEngine(solar_site_actual_hists,
                                     solar_site_forecast_hists,
                                     scenario_start_time, solar_meta_df,
                                     us_state='California')
        dist = solar_engn.asset_distance().values

        if create_load_solar:
            solar_engn.fit_load_solar_joint_model(
                load_hist_actual_df=load_zone_actual_hists,
                load_hist_forecast_df=load_zone_forecast_hists,
                asset_rho=dist / (10 * dist.max()), pca_comp_rho=5e-2,
                num_of_components=components, nearest_days=nearest_days,
                use_all_load_hist=use_all_load_hist
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

            if write_csv:
                solar_engn.write_to_csv(out_dir,
                                        {'load': load_zone_actual_futures,
                                         'solar': solar_site_actual_futures},
                                        write_forecasts=True)

            else:
                out_scens['Load'] = solar_engn.scenarios['load'].round(4)
                out_scens['Solar'] = solar_engn.scenarios['solar'].round(4)

            if verbosity >= 2:
                mdl_components = solar_engn.solar_md.num_of_components
                mdl_explained = 1 - solar_engn.solar_md.pca_residual

        else:
            solar_engn.fit(asset_rho=dist / (10 * dist.max()),
                           pca_comp_rho=5e-2, num_of_components=components,
                           nearest_days=nearest_days)
            solar_engn.create_scenario(scen_count, solar_site_forecast_futures)

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

            if write_csv:
                solar_engn.write_to_csv(out_dir,
                                        {'solar': solar_site_actual_futures},
                                        write_forecasts=True)

            else:
                out_scens['Solar'] = solar_engn.scenarios['solar'].round(4)

            if verbosity >= 2:
                mdl_components = solar_engn.model.num_of_components
                mdl_explained = 1 - solar_engn.model.pca_residual

        if not write_csv:
            with bz2.BZ2File(out_fl, 'w') as f:
                pickle.dump(out_scens, f, protocol=-1)

        if get_energy_scores:
            with bz2.BZ2File(Path(out_fl.parent,
                                  'escores_{}.p.gz'.format(date_lbl)),
                             'w') as f:
                pickle.dump(energy_scores, f, protocol=-1)

        if get_variograms:
            with bz2.BZ2File(Path(out_fl.parent,
                                  'varios_{}.p.gz'.format(date_lbl)),
                             'w') as f:
                pickle.dump(variograms, f, protocol=-1)

        if verbosity >= 2:
            print("Used {} PCA components which explained {:.2%} of the "
                  "variance in the solar training data.".format(mdl_components,
                                                                mdl_explained))

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
