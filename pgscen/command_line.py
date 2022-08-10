"""Command line interface for generating scenarios with ERCOT/NREL datasets."""

import argparse
import os
from pathlib import Path
import shutil
import bz2
import dill as pickle
import time

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Iterable

import numpy as np
import pandas as pd

from .utils.data_utils import (load_load_data, load_wind_data, load_solar_data,
                               split_actuals_hist_future,
                               split_forecasts_hist_future)

from .engine import GeminiEngine, SolarGeminiEngine
from .pca import PCAGeminiEngine, PCAGeminiModel
from .scoring import compute_energy_scores, compute_variograms


# define common command-line arguments across all tools
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

parent_parser.add_argument('--nearest-days', '-d', type=int, default=50,
                           dest='nearest_days',
                           help="the size of the historical window to use "
                                "when training")

parent_parser.add_argument('--asset-rho', type=float,
                           default=0.05, dest='asset_rho')
parent_parser.add_argument('--time-rho', type=float,
                           default=0.05, dest='time_rho')

parent_parser.add_argument('--bin-width-ratio', type=float,
                           default=0.1, dest='bin_width_ratio')
parent_parser.add_argument('--min-sample-size', type=float,
                           default=400, dest='min_sample_size')

parent_parser.add_argument('--random-seed', type=int, dest='random_seed',
                           help="fix the stochastic component of scenario "
                                "generation for testing purposes")

parent_parser.add_argument('--pickle', '-p', action='store_true',
                           help="store output in .p.gz format instead of .csv")
parent_parser.add_argument('--skip-existing',
                           action='store_true', dest='skip_existing',
                           help="don't overwrite existing output files")
parent_parser.add_argument('--verbose', '-v', action='count', default=0)

parent_parser.add_argument('--energy-scores',
                           action='store_true', dest='energy_scores',
                           help="quantify scenario quality with energy scores")
parent_parser.add_argument('--variograms', action='store_true',
                           help="quantify scenario quality with variograms")

parent_parser.add_argument('--test', action='store_true')

# add arguments to parent_parser for different tuning types
parent_parser.add_argument('--tuning', type=str, default='', dest='tuning',
                           choices=['rhos', 'nearest_days', 'wind_specific', 'components'],
                           help='string to indicate the tuning type')
parent_parser.add_argument('--tuning-list-1', action="extend", nargs="+", type=float,
                           dest='tuning_list_1', help='the list of tuning param 1')
parent_parser.add_argument('--tuning-list-2', action="extend", nargs="+", type=float,
                           dest='tuning_list_2', help='the list of tuning param 2')

test_path = Path(Path(__file__).parent.parent, 'test', 'resources')

pca_parser = argparse.ArgumentParser(add_help=False, parents=[parent_parser])
pca_parser.add_argument(
    '--components', '-c', type=str, default='0.99',
    help="How many factors to use when fitting the PCA."
         "See sklearn.decomposition.PCA for possible argument values."
)

joint_parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
joint_pca_parser = argparse.ArgumentParser(parents=[pca_parser],
                                           add_help=False)

for prs in (joint_parser, joint_pca_parser):
    prs.add_argument('--use-all-load-history',
                     action='store_true', dest='use_all_load_hist',
                     help="train load models using all out-of-sample "
                          "historical days instead of the same "
                          "window used for solar models")


# tools for creating a particular type of scenario
def create_load_scenarios():
    args = argparse.ArgumentParser(
        'pgscen-load', parents=[parent_parser],
        description="Create day ahead load scenarios."
        ).parse_args()

    scen_generator = T7kScenarioGenerator(args)
    scen_generator.produce_scenarios(create_load=True)


def create_wind_scenarios():
    args = argparse.ArgumentParser(
        'pgscen-wind', parents=[parent_parser],
        description="Create day ahead wind scenarios."
        ).parse_args()

    scen_generator = T7kScenarioGenerator(args)
    scen_generator.produce_scenarios(create_wind=True)


def create_solar_scenarios():
    args = argparse.ArgumentParser(
        'pgscen-solar', parents=[parent_parser],
        description="Create day ahead solar scenarios."
        ).parse_args()

    scen_generator = T7kScenarioGenerator(args)
    scen_generator.produce_scenarios(create_solar=True)


def create_load_solar_scenarios():
    args = argparse.ArgumentParser(
        'pgscen-load-solar', parents=[joint_parser],
        description="Create day ahead load-solar jointly modeled scenarios."
        ).parse_args()

    scen_generator = T7kScenarioGenerator(args)
    scen_generator.produce_scenarios(create_load_solar=True)


# tool for creating all types of scenarios at the same time
def create_scenarios():
    parser = argparse.ArgumentParser(
        'pgscen', parents=[joint_parser],
        description="Create day-ahead t7k load, wind, and solar scenarios."
        )

    parser.add_argument('--joint', action='store_true',
                        help="use a joint load-solar model")

    args = parser.parse_args()
    scen_generator = T7kScenarioGenerator(args)

    if args.joint:
        scen_generator.produce_scenarios(create_wind=True,
                                         create_load_solar=True)
    else:
        scen_generator.produce_scenarios(create_load=True, create_wind=True)
        scen_generator.produce_scenarios(create_solar=True)


# tools for creating scenarios using Principal Component Analysis time-features
def create_pca_solar_scenarios():
    args = argparse.ArgumentParser(
        'pgscen-pca-solar', parents=[pca_parser],
        description="Create day ahead solar scenarios using PCA features."
        ).parse_args()

    scen_generator = T7kPCAScenarioGenerator(args)
    scen_generator.produce_scenarios(create_solar=True)


def create_pca_load_solar_scenarios():
    args = argparse.ArgumentParser(
        'pgscen-pca-load-solar', parents=[joint_pca_parser],
        description="Create day ahead load-solar jointly modeled scenarios."
        ).parse_args()

    scen_generator = T7kPCAScenarioGenerator(args)
    scen_generator.produce_scenarios(create_load_solar=True)


def create_pca_scenarios():
    parser = argparse.ArgumentParser(
        'pgscen-pca', parents=[joint_pca_parser],
        description="Create day-ahead t7k load, wind, and solar PCA scenarios."
        )

    parser.add_argument('--joint', action='store_true',
                        help="use a joint load-solar model")

    args = parser.parse_args()
    scen_generator = T7kPCAScenarioGenerator(args)

    if args.joint:
        scen_generator.produce_scenarios(create_wind=True,
                                         create_load_solar=True)
    else:
        scen_generator.produce_scenarios(create_load=True, create_wind=True,
                                         create_solar=True)


class ScenarioGenerator(ABC):
    """Abstract base class for interfaces between command-line and PGscen."""

    scenario_label = None
    us_state = None
    start_hour = None

    def __init__(self, args: argparse.Namespace) -> None:
        self.start_date = args.start
        self.ndays = args.days
        self.scen_count = args.scenario_count

        self.asset_rho = args.asset_rho
        self.time_rho = args.time_rho
        self.nearest_days = args.nearest_days
        self.bin_width_ratio = args.bin_width_ratio
        self.min_sample_size = args.min_sample_size
        self.use_all_load_hist = args.use_all_load_hist

        self.output_dir = args.out_dir
        self.write_csv = not args.pickle
        self.skip_existing = args.skip_existing
        self.random_seed = args.random_seed
        self.verbosity = args.verbose
        self.start_time = ' '.join([self.start_date, self.start_hour])

        self.energy_scores = args.energy_scores
        self.variograms = args.variograms

        self.tuning = args.tuning

        self.actuals = {'load': None, 'wind': None, 'solar': None}
        self.forecasts = {'load': None, 'wind': None, 'solar': None}
        self.metadata = {'wind': None, 'solar': None}
        self.futures = {'load': None, 'wind': None, 'solar': None}

    def days(self) -> Iterable[Tuple[str, pd.DatetimeIndex, Path]]:
        """Looping over the days and times we are creating scenarios for."""

        for daily_start_time in pd.date_range(start=self.start_time,
                                              periods=self.ndays,
                                              freq='D', tz='utc'):
            date_lbl = daily_start_time.strftime('%Y-%m-%d')

            if self.write_csv:
                out_path = Path(self.output_dir,
                                daily_start_time.strftime('%Y%m%d'))

                if out_path.exists():
                    if self.skip_existing:
                        continue
                    else:
                        shutil.rmtree(out_path)

            else:
                out_path = Path(self.output_dir,
                                "scens_{}.p.gz".format(date_lbl))

                if self.skip_existing and out_path.exists():
                    continue

            scen_timesteps = pd.date_range(start=daily_start_time,
                                           periods=24, freq='H')

            if self.verbosity >= 1:
                print("Creating {} scenarios for: {}".format(
                    self.scenario_label, date_lbl))

            yield date_lbl, scen_timesteps, out_path

    def daily_message(self, scen_engines: Dict[str, GeminiEngine]) -> None:
        """Logging info about the scenario models fit for a given day."""
        pass

    def produce_scenarios_tuning(self, create_load: bool = False, create_wind: bool = False,
                                 create_solar: bool = False,
                                 create_load_solar: bool = False, asset_rho=None, time_rho=None,
                                 nearest_days=None, bin_width_ratio=None, min_sample_size=None,
                                 components=None):
        # Change the output directory to store files with different tunings in different folders
        # Update the parameter to the tuning value
        if self.tuning == 'rhos':
            self.output_dir = os.path.join(self.output_dir, self.tuning + '_' + str(asset_rho) + '_' + str(time_rho))
            if not os.path.isdir(self.output_dir):
                os.mkdir(self.output_dir)
            self.asset_rho = asset_rho
            self.time_rho = time_rho
        elif self.tuning == 'nearest_days':
            self.output_dir = os.path.join(self.output_dir, self.tuning + '_' + str(components))
            if not os.path.isdir(self.output_dir):
                os.mkdir(self.output_dir)
            self.nearest_days = nearest_days
        elif self.tuning == 'wind_specific':
            self.output_dir = os.path.join(self.output_dir, self.tuning + '_'
                                           + str(bin_width_ratio) + '_' + str(min_sample_size))
            if not os.path.isdir(self.output_dir):
                os.mkdir(self.output_dir)
            self.bin_width_ratio = bin_width_ratio
            self.min_sample_size = min_sample_size
        elif self.tuning == 'components':
            self.output_dir = os.path.join(self.output_dir, self.tuning + '_' + str(components))
            if not os.path.isdir(self.output_dir):
                os.mkdir(self.output_dir)
            self.components = components

        self.produce_scenarios(create_load, create_wind, create_solar, create_load_solar)

    def produce_scenarios(self,
                          create_load: bool = False, create_wind: bool = False,
                          create_solar: bool = False,
                          create_load_solar: bool = False) -> None:
        """Generates given types of scenarios for all days."""

        if create_load_solar and (create_load or create_solar):
            raise ValueError("Cannot create load-solar joint models at the "
                             "same time as load-only or solar-only models!")

        if self.random_seed:
            np.random.seed(self.random_seed)

        if self.verbosity >= 2:
            t0 = time.time()

        for date_lbl, scen_timesteps, out_path in self.days():
            scen_engines = dict()

            out_scens = dict()
            energy_scores = dict()
            variograms = dict()

            if create_load:
                scen_engines['load'] = self.produce_load_scenarios(
                    scen_timesteps)

            if create_wind:
                scen_engines['wind'] = self.produce_wind_scenarios(
                    scen_timesteps)

            if create_solar:
                scen_engines['solar'] = self.produce_solar_scenarios(
                    scen_timesteps)

            if create_load_solar:
                scen_engines['load'] = self.produce_load_solar_scenarios(
                    scen_timesteps)
                scen_engines['solar'] = scen_engines['load']

            for asset_type, scen_engine in scen_engines.items():
                asset_lbl = asset_type.capitalize()

                if self.energy_scores:
                    energy_scores[asset_lbl] = compute_energy_scores(
                        scen_engine.scenarios[asset_type],
                        self.actuals[asset_type], self.forecasts[asset_type]
                    )

                if self.variograms:
                    variograms[asset_lbl] = compute_variograms(
                        scen_engine.scenarios[asset_type],
                        self.actuals[asset_type], self.forecasts[asset_type]
                    )

                if self.tuning == '':
                    if self.write_csv:
                        scen_engine.write_to_csv(self.output_dir,
                                                 self.futures[asset_type],
                                                 write_forecasts=True)
                    else:
                        out_scens[asset_type.capitalize()] = scen_engine.scenarios[
                            asset_type].round(4)

            if self.tuning == '' and not self.write_csv:
                with bz2.BZ2File(out_path, 'w') as f:
                    pickle.dump(out_scens, f, protocol=-1)

            # Update output path based on tuning or not
            if self.tuning == '':
                scores_out_path = out_path.parent
            else:
                scores_out_path = self.output_dir

            if self.energy_scores:
                with bz2.BZ2File(Path(scores_out_path,
                                      'escores_{}.p.gz'.format(date_lbl)),
                                 'w') as f:
                    pickle.dump(energy_scores, f, protocol=-1)

            if self.variograms:
                with bz2.BZ2File(Path(scores_out_path,
                                      'varios_{}.p.gz'.format(date_lbl)),
                                 'w') as f:
                    pickle.dump(variograms, f, protocol=-1)

            self.daily_message(scen_engines)

        if self.verbosity >= 2:
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

            if self.ndays == 1:
                day_str = "day"
            else:
                day_str = "days"

            print("Created {} {} scenarios for {} {} in {:.1f} "
                  "seconds".format(self.scen_count, type_str, self.ndays,
                                   day_str, time.time() - t0))

    @abstractmethod
    def produce_load_scenarios(
            self, scen_timesteps: pd.DatetimeIndex) -> GeminiEngine:
        pass

    @abstractmethod
    def produce_wind_scenarios(
            self, scen_timesteps: pd.DatetimeIndex) -> GeminiEngine:
        pass

    @abstractmethod
    def produce_solar_scenarios(
            self, scen_timesteps: pd.DatetimeIndex) -> SolarGeminiEngine:
        pass

    @abstractmethod
    def produce_load_solar_scenarios(
            self, scen_timesteps: pd.DatetimeIndex) -> SolarGeminiEngine:
        pass


class PCAScenarioGenerator(ScenarioGenerator, ABC):
    """Abstract class for creating PCA solar scenarios via command-line."""

    def __init__(self, args: argparse.Namespace) -> None:
        if args.components.isdigit() and args.components != '0':
            self.components = int(args.components)

        elif args.components[:2] == '0.' and args.components[2:].isdigit():
            self.components = float(args.components)
        elif args.components[0] == '.' and args.components[1:].isdigit():
            self.components = float(args.components)

        elif args.components != 'mle':
            raise ValueError("Invalid <components> value of `{}`! See "
                             "sklearn.decomposition.PCA for accepted argument "
                             "values.".format(self.components))

        super().__init__(args)

    def daily_message(self, scen_engines: Dict[str, GeminiEngine]) -> None:
        """Prints how much of the solar variance the PCA model explained."""

        if isinstance(scen_engines['solar'].model, PCAGeminiModel):
            use_model = scen_engines['solar'].model
        else:
            use_model = scen_engines['solar'].solar_md

        if self.verbosity >= 2 and 'solar' in scen_engines:
            print("Used {} PCA components which explained "
                  "{:.2%} of the variance in the solar "
                  "training data.".format(use_model.num_of_components,
                                          1 - use_model.pca_residual))

    @abstractmethod
    def produce_solar_scenarios(
            self, scen_timesteps: pd.DatetimeIndex) -> PCAGeminiEngine:
        pass

    @abstractmethod
    def produce_load_solar_scenarios(
            self, scen_timesteps: pd.DatetimeIndex) -> PCAGeminiEngine:
        pass


class T7kScenarioGenerator(ScenarioGenerator):
    """Class used by command-line tools for creating Texas-7k scenarios."""

    scenario_label = "Texas-7k"
    us_state = 'Texas'
    start_hour = '06:00:00'

    def produce_load_scenarios(
            self, scen_timesteps: pd.DatetimeIndex) -> GeminiEngine:

        if self.actuals['load'] is None:
            self.actuals['load'], self.forecasts['load'] = load_load_data()

        # split input datasets into training and testing subsets
        # for RTS we always do in-sample since we only have a year of data
        (load_zone_actual_hists,
            self.futures['load']) = split_actuals_hist_future(
                    self.actuals['load'], scen_timesteps, in_sample=False)
        (load_zone_forecast_hists,
            load_zone_forecast_futures) = split_forecasts_hist_future(
                    self.forecasts['load'], scen_timesteps, in_sample=False)

        load_engn = GeminiEngine(load_zone_actual_hists,
                                 load_zone_forecast_hists,
                                 scen_timesteps[0], asset_type='load')

        load_engn.fit(self.asset_rho, self.time_rho)
        load_engn.create_scenario(self.scen_count,
                                  load_zone_forecast_futures,
                                  bin_width_ratio=self.bin_width_ratio, min_sample_size=self.min_sample_size)

        return load_engn

    def produce_wind_scenarios(
            self, scen_timesteps: pd.DatetimeIndex) -> GeminiEngine:

        if self.actuals['wind'] is None:
            (self.actuals['wind'], self.forecasts['wind'],
                self.metadata['wind']) = load_wind_data()

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
                self.metadata['solar']) = load_solar_data()

        (solar_site_actual_hists,
            self.futures['solar']) = split_actuals_hist_future(
                    self.actuals['solar'], scen_timesteps, in_sample=False)
        (solar_site_forecast_hists,
            solar_site_forecast_futures) = split_forecasts_hist_future(
                    self.forecasts['solar'], scen_timesteps, in_sample=False)

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
            self.actuals['load'], self.forecasts['load'] = load_load_data()

        if self.actuals['solar'] is None:
            (self.actuals['solar'], self.forecasts['solar'],
                self.metadata['solar']) = load_solar_data()

        # split input datasets into training and testing subsets
        # for RTS we always do in-sample since we only have a year of data
        (load_zone_actual_hists,
            self.futures['load']) = split_actuals_hist_future(
                    self.actuals['load'], scen_timesteps, in_sample=False)
        (load_zone_forecast_hists,
            load_zone_forecast_futures) = split_forecasts_hist_future(
                    self.forecasts['load'], scen_timesteps, in_sample=False)

        (solar_site_actual_hists,
            self.futures['solar']) = split_actuals_hist_future(
                    self.actuals['solar'], scen_timesteps, in_sample=False)
        (solar_site_forecast_hists,
            solar_site_forecast_futures) = split_forecasts_hist_future(
                    self.forecasts['solar'], scen_timesteps, in_sample=False)

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


class T7kPCAScenarioGenerator(PCAScenarioGenerator, T7kScenarioGenerator):
    """Class used by command-line tools for creating Texas-7k PCA scenarios."""

    scenario_label = "Texas-7k PCA"

    def produce_solar_scenarios(
            self, scen_timesteps: pd.DatetimeIndex) -> PCAGeminiEngine:

        if self.actuals['solar'] is None:
            (self.actuals['solar'], self.forecasts['solar'],
                self.metadata['solar']) = load_solar_data()

        (solar_site_actual_hists,
            self.futures['solar']) = split_actuals_hist_future(
                    self.actuals['solar'], scen_timesteps, in_sample=False)
        (solar_site_forecast_hists,
            solar_site_forecast_futures) = split_forecasts_hist_future(
                    self.forecasts['solar'], scen_timesteps, in_sample=False)

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
            self.actuals['load'], self.forecasts['load'] = load_load_data()

        if self.actuals['solar'] is None:
            (self.actuals['solar'], self.forecasts['solar'],
                self.metadata['solar']) = load_solar_data()

        # split input datasets into training and testing subsets
        # for RTS we always do in-sample since we only have a year of data
        (load_zone_actual_hists,
            self.futures['load']) = split_actuals_hist_future(
                    self.actuals['load'], scen_timesteps, in_sample=False)
        (load_zone_forecast_hists,
            load_zone_forecast_futures) = split_forecasts_hist_future(
                    self.forecasts['load'], scen_timesteps, in_sample=False)

        (solar_site_actual_hists,
            self.futures['solar']) = split_actuals_hist_future(
                    self.actuals['solar'], scen_timesteps, in_sample=False)
        (solar_site_forecast_hists,
            solar_site_forecast_futures) = split_forecasts_hist_future(
                    self.forecasts['solar'], scen_timesteps, in_sample=False)

        solar_engn = PCAGeminiEngine(solar_site_actual_hists,
                                     solar_site_forecast_hists,
                                     scen_timesteps[0], self.metadata['solar'],
                                     us_state=self.us_state)

        dist = solar_engn.asset_distance().values
        solar_asset_rho = 2 * self.asset_rho * dist / dist.max()

        #TODO: double-check these regularization parametrizations
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
