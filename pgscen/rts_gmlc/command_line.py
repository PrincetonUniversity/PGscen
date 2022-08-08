"""Command line interface for generating scenarios for the RTS-GMLC system."""

import argparse
from pathlib import Path
import shutil
import time
import bz2
import dill as pickle

import numpy as np
import pandas as pd

from ..command_line import parent_parser
from ..pca_command_line import pca_parser
from ..engine import GeminiEngine, SolarGeminiEngine
from ..pca import PCAGeminiEngine
from ..scoring import compute_energy_scores, compute_variograms

from .data_utils import load_load_data, load_wind_data, load_solar_data
from ..utils.data_utils import (split_actuals_hist_future,
                                split_forecasts_hist_future)


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


#TODO: RTS solar models seem to be numerically unstable — why?
def create_scenarios():
    args = argparse.ArgumentParser(
        'pgscen-rts', parents=[rts_parser],
        description="Create day-ahead RTS load, wind, and solar scenarios."
        ).parse_args()

    scen_generator = ScenarioGenerator(args)
    scen_generator.produce_scenarios(create_load=True, create_wind=True,
                                     create_solar=True)


def create_joint_scenarios():
    args = argparse.ArgumentParser(
        'pgscen-rts-joint', parents=[joint_parser],
        description="Create day-ahead RTS wind and load-solar joint scenarios."
        ).parse_args()

    scen_generator = ScenarioGenerator(args)
    scen_generator.produce_scenarios(create_wind=True, create_load_solar=True)


def create_pca_solar_scenarios():
    args = argparse.ArgumentParser(
        'pgscen-rts-pca-solar', parents=[rts_pca_parser],
        description="Create day ahead RTS solar scenarios using PCA features."
        ).parse_args()

    scen_generator = PCAScenarioGenerator(args)
    scen_generator.produce_scenarios(create_solar=True)

def create_pca_load_solar_scenarios():
    args = argparse.ArgumentParser(
        'pgscen-rts-pca-load-solar', parents=[joint_pca_parser],
        description="Create day ahead RTS load-solar joint-modeled scenarios."
        ).parse_args()

    scen_generator = PCAScenarioGenerator(args)
    scen_generator.produce_scenarios(create_load_solar=True)


def create_pca_scenarios():
    parser = argparse.ArgumentParser(
        'pgscen-rts-pca', parents=[joint_pca_parser],
        description="Create day-ahead RTS load, wind, and solar PCA scenarios."
        )

    parser.add_argument('--joint', action='store_true',
                        help="use a joint load-solar model")

    args = parser.parse_args()
    scen_generator = PCAScenarioGenerator(args)

    if args.joint:
        scen_generator.produce_scenarios(create_wind=True,
                                         create_load_solar=True)
    else:
        scen_generator.produce_scenarios(create_load=True, create_wind=True)
        scen_generator.produce_scenarios(create_solar=True)


class ScenarioGenerator:

    scenario_label = "RTS-GMLC"

    def __init__(self, args):
        self.start_date = args.start
        self.ndays = args.days
        self.scen_count = args.scenario_count

        self.asset_rho = args.asset_rho
        self.time_rho = args.time_rho
        self.nearest_days = args.nearest_days
        self.use_all_load_hist = args.use_all_load_hist

        self.input_dir = args.rts_dir
        self.output_dir = args.out_dir
        self.write_csv = not args.pickle
        self.skip_existing = args.skip_existing
        self.random_seed = args.random_seed
        self.verbosity = args.verbose

        self.start_time = ' '.join([self.start_date, '08:00:00'])
        self.us_state = 'California'

        self.actuals = {'load': None, 'wind': None, 'solar': None}
        self.forecasts = {'load': None, 'wind': None, 'solar': None}
        self.metadata = {'wind': None, 'solar': None}
        self.futures = {'load': None, 'wind': None, 'solar': None}

    def days(self):
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

    def daily_message(self, scen_engines):
        pass

    def produce_scenarios(self,
                          create_load=False, create_wind=False,
                          create_solar=False, create_load_solar=False,
                          get_energy_scores=False, get_variograms=False):
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

                if get_energy_scores:
                    energy_scores[asset_lbl] = compute_energy_scores(
                        scen_engine.scenarios[asset_type],
                        self.actuals[asset_type], self.forecasts[asset_type]
                        )

                if get_variograms:
                    variograms[asset_lbl] = compute_variograms(
                        scen_engine.scenarios[asset_type],
                        self.actuals[asset_type], self.forecasts[asset_type]
                        )

                if self.write_csv:
                    scen_engine.write_to_csv(self.output_dir,
                                             self.futures[asset_type],
                                             write_forecasts=True)
                else:
                    out_scens[asset_type.capitalize()] = scen_engine.scenarios[
                        asset_type].round(4)

            if not self.write_csv:
                with bz2.BZ2File(out_path, 'w') as f:
                    pickle.dump(out_scens, f, protocol=-1)

            if get_energy_scores:
                with bz2.BZ2File(Path(out_path.parent,
                                      'escores_{}.p.gz'.format(date_lbl)),
                                 'w') as f:
                    pickle.dump(energy_scores, f, protocol=-1)

            if get_variograms:
                with bz2.BZ2File(Path(out_path.parent,
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

    def produce_load_scenarios(self, scen_timesteps):
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

    def produce_wind_scenarios(self, scen_timesteps):
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

    def produce_solar_scenarios(self, scen_timesteps):
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

    def produce_load_solar_scenarios(self, scen_timesteps):
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


class PCAScenarioGenerator(ScenarioGenerator):

    scenario_label = "RTS-GMLC PCA"

    def __init__(self, args):
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

    def daily_message(self, scen_engines):
        """Prints how much of the solar variance the PCA model explained."""

        if self.verbosity >= 2 and 'solar' in scen_engines:
            mdl_components = scen_engines['solar'].model.num_of_components
            mdl_explained = 1 - scen_engines['solar'].model.pca_residual

            print("Used {} PCA components which explained "
                  "{:.2%} of the variance in the solar "
                  "training data.".format(mdl_components, mdl_explained))

    def produce_solar_scenarios(self, scen_timesteps):
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

    def produce_load_solar_scenarios(self, scen_timesteps):
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

        solar_engn.fit_load_solar_joint_model(
            load_hist_actual_df=load_zone_actual_hists,
            load_hist_forecast_df=load_zone_forecast_hists,
            joint_asset_rho=dist / (10 * dist.max()),
            solar_pca_comp_rho=5e-2, num_of_components=self.components,
            nearest_days=self.nearest_days,
            use_all_load_hist=self.use_all_load_hist
            )

        solar_engn.create_load_solar_joint_scenario(
            self.scen_count,
            load_zone_forecast_futures, solar_site_forecast_futures
            )

        return solar_engn
