"""Command line interface for generating scenarios for the RTS-GMLC system."""

import argparse
import pandas as pd

from ..command_line import (parent_parser, pca_parser,
                            ScenarioGenerator, PCAScenarioGenerator)

from ..engine import GeminiEngine, SolarGeminiEngine
from ..pca import PCAGeminiEngine

from .data_utils import load_load_data, load_wind_data, load_solar_data
from ..utils.data_utils import (split_actuals_hist_future,
                                split_forecasts_hist_future)
from joblib import Parallel, delayed


rts_parser = argparse.ArgumentParser(add_help=False, parents=[parent_parser])
rts_pca_parser = argparse.ArgumentParser(add_help=False, parents=[pca_parser])

for parser in (rts_parser, rts_pca_parser):
    parser.add_argument('rts_dir', type=str,
                        help="where RTS-GMLC repository is stored")

joint_parser = argparse.ArgumentParser(parents=[rts_parser], add_help=False)
joint_pca_parser = argparse.ArgumentParser(parents=[rts_pca_parser],
                                           add_help=False)

for parser in (joint_parser, joint_pca_parser):
    parser.add_argument('--use-all-load-history',
                        action='store_true', dest='use_all_load_hist',
                        help="train load models using all out-of-sample "
                             "historical days instead of the same "
                             "window used for solar models")


# TODO: RTS solar models seem to be numerically unstable — why?
def create_scenarios():
    args = argparse.ArgumentParser(
        'pgscen-rts', parents=[rts_parser],
        description="Create day-ahead RTS load, wind, and solar scenarios."
    ).parse_args()

    scen_generator = RtsScenarioGenerator(args)
    if args.tuning == '':
        scen_generator.produce_scenarios(create_load=True, create_wind=True,
                                         create_solar=True)
    elif args.tuning == 'rhos':
        Parallel(n_jobs=31, verbose=-1)(
            delayed(scen_generator.produce_scenarios_tuning)(create_load=True, create_wind=True,
                                                             create_solar=True, asset_rho=asset_rho, time_rho=time_rho)
            for asset_rho in args.tuning_list_1 for time_rho in args.tuning_list_2)

    # neraest days is only used in the solar scenarios
    elif args.tuning == 'nearest_days':
        Parallel(n_jobs=31, verbose=-1)(
            delayed(scen_generator.produce_scenarios_tuning)(create_load=False, create_wind=False,
                                                             create_solar=True, nearest_days=nearest_days)
            for nearest_days in args.tuning_list_1)
    # wind specific is only used in the wind scenarios
    elif args.tuning == 'wind_specific':
        Parallel(n_jobs=31, verbose=-1)(
            delayed(scen_generator.produce_scenarios_tuning)(create_load=False, create_wind=True,
                                                             create_solar=False,
                                                             bin_width_ratio=bin_width_ratio,
                                                             min_sample_size=min_sample_size)
            for bin_width_ratio in args.tuning_list_1 for min_sample_size in args.tuning_list_2)


def create_joint_scenarios():
    args = argparse.ArgumentParser(
        'pgscen-rts-joint', parents=[joint_parser],
        description="Create day-ahead RTS wind and load-solar joint scenarios."
    ).parse_args()

    scen_generator = RtsScenarioGenerator(args)
    scen_generator.produce_scenarios(create_wind=True, create_load_solar=True)


def create_pca_solar_scenarios():
    args = argparse.ArgumentParser(
        'pgscen-rts-pca-solar', parents=[rts_pca_parser],
        description="Create day ahead RTS solar scenarios using PCA features."
    ).parse_args()

    scen_generator = RtsPCAScenarioGenerator(args)
    if args.tuning == 'components':
        Parallel(n_jobs=31, verbose=-1)(
            delayed(scen_generator.produce_scenarios_tuning)(create_solar=True,
                                                             components=components)
            for components in args.tuning_list_1)


def create_pca_load_solar_scenarios():
    args = argparse.ArgumentParser(
        'pgscen-rts-pca-load-solar', parents=[joint_pca_parser],
        description="Create day ahead RTS load-solar joint-modeled scenarios."
        ).parse_args()

    scen_generator = RtsPCAScenarioGenerator(args)
    scen_generator.produce_scenarios(create_load_solar=True)


def create_pca_scenarios():
    parser = argparse.ArgumentParser(
        'pgscen-rts-pca', parents=[joint_pca_parser],
        description="Create day-ahead RTS load, wind, and solar PCA scenarios."
        )

    parser.add_argument('--joint', action='store_true',
                        help="use a joint load-solar model")

    args = parser.parse_args()
    scen_generator = RtsPCAScenarioGenerator(args)

    if args.joint:
        scen_generator.produce_scenarios(create_wind=True,
                                         create_load_solar=True)
    else:
        scen_generator.produce_scenarios(create_load=True, create_wind=True,
                                         create_solar=True)


class RtsScenarioGenerator(ScenarioGenerator):
    """Class used by command-line tools for creating RTS-GMLC scenarios."""

    scenario_label = "RTS-GMLC"
    us_state = 'California'
    start_hour = '08:00:00'

    def __init__(self, args):
        self.input_dir = args.rts_dir

        super().__init__(args)

    def produce_load_scenarios(
            self, scen_timesteps: pd.DatetimeIndex) -> GeminiEngine:

        if self.actuals['load'] is None:
            self.actuals['load'], self.forecasts['load'] = load_load_data(
                self.input_dir)

        # split input datasets into training and testing subsets
        # for RTS we always do in-sample since we only have a year of data
        (load_zone_actual_hists,
            self.futures['load']) = split_actuals_hist_future(
                    self.actuals['load'], scen_timesteps, in_sample=True)
        (load_zone_forecast_hists,
            load_zone_forecast_futures) = split_forecasts_hist_future(
                    self.forecasts['load'], scen_timesteps, in_sample=True)

        load_engn = GeminiEngine(load_zone_actual_hists,
                                 load_zone_forecast_hists,
                                 scen_timesteps[0], asset_type='load')

        load_engn.fit(self.asset_rho, self.time_rho)
        load_engn.create_scenario(self.scen_count,
                                  load_zone_forecast_futures,
                                  bin_width_ratio=0.1, min_sample_size=400)

        return load_engn

    def produce_wind_scenarios(
            self, scen_timesteps: pd.DatetimeIndex) -> GeminiEngine:

        if self.actuals['wind'] is None:
            (self.actuals['wind'], self.forecasts['wind'],
                self.metadata['wind']) = load_wind_data(self.input_dir)

        (wind_site_actual_hists,
            self.futures['wind']) = split_actuals_hist_future(
                    self.actuals['wind'], scen_timesteps, in_sample=True)
        (wind_site_forecast_hists,
            wind_site_forecast_futures) = split_forecasts_hist_future(
                    self.forecasts['wind'], scen_timesteps, in_sample=True)

        wind_engn = GeminiEngine(
            wind_site_actual_hists, wind_site_forecast_hists,
            scen_timesteps[0], self.metadata['wind'], asset_type='wind'
            )

        dist = wind_engn.asset_distance().values
        wind_engn.fit(2 * self.asset_rho * dist / dist.max(), self.time_rho)
        wind_engn.create_scenario(self.scen_count, wind_site_forecast_futures)

        return wind_engn

    def produce_solar_scenarios(
            self, scen_timesteps: pd.DatetimeIndex) -> SolarGeminiEngine:

        if self.actuals['solar'] is None:
            (self.actuals['solar'], self.forecasts['solar'],
                self.metadata['solar']) = load_solar_data(self.input_dir)

        (solar_site_actual_hists,
            self.futures['solar']) = split_actuals_hist_future(
                    self.actuals['solar'], scen_timesteps, in_sample=True)
        (solar_site_forecast_hists,
            solar_site_forecast_futures) = split_forecasts_hist_future(
                    self.forecasts['solar'], scen_timesteps, in_sample=True)

        solar_engn = SolarGeminiEngine(
            solar_site_actual_hists, solar_site_forecast_hists,
            scen_timesteps[0], self.metadata['solar'], us_state=self.us_state
            )

        solar_engn.fit_solar_model(nearest_days=self.nearest_days)
        solar_engn.create_solar_scenario(self.scen_count,
                                         solar_site_forecast_futures)

        return solar_engn

    def produce_load_solar_scenarios(
            self, scen_timesteps: pd.DatetimeIndex) -> SolarGeminiEngine:

        if self.actuals['load'] is None:
            self.actuals['load'], self.forecasts['load'] = load_load_data(
                self.input_dir)

        if self.actuals['solar'] is None:
            (self.actuals['solar'], self.forecasts['solar'],
                self.metadata['solar']) = load_solar_data(self.input_dir)

        # split input datasets into training and testing subsets
        # for RTS we always do in-sample since we only have a year of data
        (load_zone_actual_hists,
            self.futures['load']) = split_actuals_hist_future(
                    self.actuals['load'], scen_timesteps, in_sample=True)
        (load_zone_forecast_hists,
            load_zone_forecast_futures) = split_forecasts_hist_future(
                    self.forecasts['load'], scen_timesteps, in_sample=True)

        (solar_site_actual_hists,
            self.futures['solar']) = split_actuals_hist_future(
                    self.actuals['solar'], scen_timesteps, in_sample=True)
        (solar_site_forecast_hists,
            solar_site_forecast_futures) = split_forecasts_hist_future(
                    self.forecasts['solar'], scen_timesteps, in_sample=True)

        solar_engn = SolarGeminiEngine(
            solar_site_actual_hists, solar_site_forecast_hists,
            scen_timesteps[0], self.metadata['solar'], us_state=self.us_state
            )

        solar_engn.fit_load_solar_joint_model(
            load_zone_actual_hists, load_zone_forecast_hists,
            nearest_days=self.nearest_days,
            use_all_load_hist=self.use_all_load_hist
            )

        solar_engn.create_load_solar_joint_scenario(
            self.scen_count,
            load_zone_forecast_futures, solar_site_forecast_futures
            )

        return solar_engn


class RtsPCAScenarioGenerator(PCAScenarioGenerator, RtsScenarioGenerator):
    """Class used by command-line tools for creating RTS-GMLC PCA scenarios."""

    scenario_label = "RTS-GMLC PCA"

    def produce_solar_scenarios(
            self, scen_timesteps: pd.DatetimeIndex) -> PCAGeminiEngine:

        if self.actuals['solar'] is None:
            (self.actuals['solar'], self.forecasts['solar'],
                self.metadata['solar']) = load_solar_data(self.input_dir)

        (solar_site_actual_hists,
            self.futures['solar']) = split_actuals_hist_future(
                    self.actuals['solar'], scen_timesteps, in_sample=True)
        (solar_site_forecast_hists,
            solar_site_forecast_futures) = split_forecasts_hist_future(
                    self.forecasts['solar'], scen_timesteps, in_sample=True)

        solar_engn = PCAGeminiEngine(solar_site_actual_hists,
                                     solar_site_forecast_hists,
                                     scen_timesteps[0], self.metadata['solar'],
                                     us_state=self.us_state)
        dist = solar_engn.asset_distance().values

        solar_engn.fit(asset_rho=2 * self.asset_rho * dist / dist.max(),
                       pca_comp_rho=self.time_rho,
                       num_of_components=self.components,
                       nearest_days=self.nearest_days)
        solar_engn.create_scenario(self.scen_count,
                                   solar_site_forecast_futures)

        return solar_engn

    def produce_load_solar_scenarios(
            self, scen_timesteps: pd.DatetimeIndex) -> PCAGeminiEngine:

        if self.actuals['load'] is None:
            self.actuals['load'], self.forecasts['load'] = load_load_data(
                self.input_dir)

        if self.actuals['solar'] is None:
            (self.actuals['solar'], self.forecasts['solar'],
                self.metadata['solar']) = load_solar_data(self.input_dir)

        # split input datasets into training and testing subsets
        # for RTS we always do in-sample since we only have a year of data
        (load_zone_actual_hists,
            self.futures['load']) = split_actuals_hist_future(
                    self.actuals['load'], scen_timesteps, in_sample=True)
        (load_zone_forecast_hists,
            load_zone_forecast_futures) = split_forecasts_hist_future(
                    self.forecasts['load'], scen_timesteps, in_sample=True)

        (solar_site_actual_hists,
            self.futures['solar']) = split_actuals_hist_future(
                    self.actuals['solar'], scen_timesteps, in_sample=True)
        (solar_site_forecast_hists,
            solar_site_forecast_futures) = split_forecasts_hist_future(
                    self.forecasts['solar'], scen_timesteps, in_sample=True)

        solar_engn = PCAGeminiEngine(solar_site_actual_hists,
                                     solar_site_forecast_hists,
                                     scen_timesteps[0], self.metadata['solar'],
                                     us_state=self.us_state)

        dist = solar_engn.asset_distance().values
        solar_asset_rho = 2 * self.asset_rho * dist / dist.max()

        solar_engn.fit_load_solar_joint_model(
            load_hist_actual_df=load_zone_actual_hists,
            load_hist_forecast_df=load_zone_forecast_hists,
            load_asset_rho=self.asset_rho, load_horizon_rho=self.time_rho,
            solar_asset_rho=solar_asset_rho, solar_pca_comp_rho=self.time_rho,
            joint_asset_rho=self.asset_rho, num_of_components=self.components,
            nearest_days=self.nearest_days,
            use_all_load_hist=self.use_all_load_hist
            )

        solar_engn.create_load_solar_joint_scenario(
            self.scen_count,
            load_zone_forecast_futures, solar_site_forecast_futures
            )

        return solar_engn
