"""Command line interface for generating scenarios for the RTS-GMLC system."""

import argparse
import pandas as pd

from ..command_line import parent_parser, pca_parser, ScenarioGenerator

from ..engine import GeminiEngine
from ..pca import PCAGeminiEngine, PCAGeminiModel

from .data_utils import load_load_data, load_wind_data, load_solar_data
from ..utils.data_utils import split_actuals_hist_future, split_forecasts_hist_future


rts_parser = argparse.ArgumentParser(add_help=False, parents=[parent_parser])
rts_pca_parser = argparse.ArgumentParser(add_help=False, parents=[pca_parser])

for parser in (rts_parser, rts_pca_parser):
    parser.add_argument('rts_dir', type=str,
                        help="where RTS-GMLC repository is stored")

def create_scenarios():
    args = argparse.ArgumentParser(
        'pgscen-rts', parents=[rts_pca_parser],
        description="Create day-ahead RTS load, wind, and solar scenarios."
        ).parse_args()

    scen_generator = RtsScenarioGenerator(args)
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
                                  load_zone_forecast_futures)

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
        pass