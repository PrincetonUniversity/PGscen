"""Engines for generating power grid asset scenarios using historical data."""

import os
from pathlib import Path
from typing_extensions import assert_type
import numpy as np
import pandas as pd

from datetime import datetime
from dateutil.relativedelta import relativedelta
import calendar
from operator import itemgetter
from astral import Observer
from astral.sun import sun
from typing import List, Dict, Set, Iterable, Optional, Union

from .model import get_asset_list, GeminiModel, GeminiError


class GeminiEngine:
    """
    A class for generating scenarios using asset actuals and forecasts.

    Attributes
    ----------
        asset_list : List[str]
            The assets for which scenarios will be generated.
        num_of_assets : int
            How many assets there are.

        hist_actual_df, hist_forecast_df : pd.DataFrame
            Historical actual and forecasted values for the assets.
        scen_start_time : pd.TimeStamp
            When the generated scenarios will start.
        meta_df : Optional[pd.DataFrame]
            Information about asset properties such as location and capacity.
        asset_type : Optional[str]
            The type of asset scenarios are being generated for.
            Must be one of 'load', 'wind', or 'solar'.
        model : Optional[GeminiModel]
            The model fitted on historical data that will be used to generate
            scenarios.

        forecast_resolution_in_minute : int
            The frequency of the intervals at which forecasts are provided.
        num_of_horizons : int
            How many forecast time intervals to generate scenarios for.
        forecast_lead_hours : int
            The time gap between when forecasts are issued and the beginning of
            the period which they are predicting.

        forecast_issue_time : pd.Timestamp
            When forecasts were issued for the period for which scenarios will
            be generated. E.g. if scenarios are to be generated for 2020-06-03,
            this might be 2020-06-02 12:00:00.
        scen_end_time : pd.Timestamp
            The end of the period for which scenarios will be generated.
        scen_timesteps : List[pd.Timestamp]
            The time points which generated scenarios will provide values for.

        forecasts : Optiona[Dict[pd.Series]]
            The forecasted values for the scenario time window which were used
            as a basis to generate scenarios.
        scenarios : Optional[Dict[pd.DataFrame]]
            The scenarios generated using this engine.
    """

    def __init__(self,
                 hist_actual_df: pd.DataFrame, hist_forecast_df: pd.DataFrame,
                 scen_start_time: pd.Timestamp,
                 meta_df: Optional[pd.DataFrame] = None,
                 asset_type: Optional[str] = None,
                 forecast_resolution_in_minute: int = 60,
                 num_of_horizons: int = 24,
                 forecast_lead_time_in_hour: int = 12) -> None:

        # check that the dataframes with actual and forecast values are in the
        # right format, get the names of the assets they contain values for
        self.asset_list = get_asset_list(hist_actual_df, hist_forecast_df)
        self.num_of_assets = len(self.asset_list)

        self.hist_actual_df = hist_actual_df
        self.hist_forecast_df = hist_forecast_df
        self.scen_start_time = scen_start_time
        self.hist_start = self.hist_forecast_df.Forecast_time.min()
        self.hist_end = self.hist_forecast_df.Forecast_time.max()

        self.meta_df = meta_df
        self.asset_type = asset_type
        self.model = None

        # standardize the format of the meta-information dataframe
        if meta_df is not None:

            # solar case
            if 'site_ids' in meta_df.columns:
                self.meta_df = self.meta_df.sort_values(
                    'site_ids').set_index('site_ids',
                                          verify_integrity=True).rename(
                        columns={'AC_capacity_MW': 'Capacity'})

            # wind case
            elif 'Facility.Name' in meta_df.columns:
                self.meta_df = self.meta_df.sort_values(
                    'Facility.Name').set_index('Facility.Name',
                                               verify_integrity=True).rename(
                        columns={'lati': 'latitude', 'longi': 'longitude'})

            else:
                raise GeminiError("Unrecognized type of metadata!")

            self.meta_df = self.meta_df[
                self.meta_df.index.isin(self.asset_list)]

        self.forecast_resolution_in_minute = forecast_resolution_in_minute
        self.num_of_horizons = num_of_horizons
        self.forecast_lead_hours = forecast_lead_time_in_hour

        # figure out when forecasts for the time period for which scenarios
        # will be generated were issued
        self.forecast_issue_time = (
            self.scen_start_time - pd.Timedelta(self.forecast_lead_hours,
                                                unit='H')
            )

        # calculate the close of the window for which scenarios will be made
        self.scen_end_time = (
                self.scen_start_time
                + pd.Timedelta((self.num_of_horizons - 1)
                               * self.forecast_resolution_in_minute,
                               unit='min')
            )

        # get the time points at which scenario values will be generated
        self.scen_timesteps = pd.date_range(
            start=self.scen_start_time, end=self.scen_end_time,
            freq=str(self.forecast_resolution_in_minute) + 'min'
            ).tolist()

        self.solar_zone_mean = None
        self.solar_zone_std = None

        self.forecasts = dict()
        self.scenarios = dict()

    def fit(self,
            asset_rho: float, horizon_rho: float,
            nearest_days: Optional[int] = None) -> None:
        """
        This function creates and fits a scenario model using historical asset
        values. The model will estimate the distributions of the deviations
        from actual values observed in the forecast dataset.

        Arguments
        ---------
            asset_rho
                Hyper-parameter governing how strongly non-zero interactions
                between generators are penalized.
            horizon_rho
                Hyper-parameter governing how strongly non-zero interactions
                between time points are penalized.

            nearest_days
                If given, will not use historical asset values more than this
                number of days away from the given date in each year.
        """

        if nearest_days:
            dev_index = self.get_yearly_date_range(use_date=self.scen_start_time,
                                              num_of_days=nearest_days)
        else:
            dev_index = None

        ## use gpd for load only
        use_gpd = True if self.asset_type == 'load' else False

        self.model = GeminiModel(self.scen_start_time, self.get_hist_df_dict(),
                                 None, dev_index,
                                 self.forecast_resolution_in_minute,
                                 self.num_of_horizons,
                                 self.forecast_lead_hours, use_gpd=use_gpd)

        self.model.fit(asset_rho, horizon_rho)

    def create_scenario(self,
                        nscen: int, forecast_df: pd.DataFrame,
                        **gpd_args) -> None:
        """
        This function generates a number of scenarios using the given forecasts
        and the model that has been fit on historical asset values.

        Arguments
        ---------
            nscen
                How many scenarios to generate.
            forecast_df
                Forecasted asset values that will be added to the deviations
                generated by the model to produce scenarios.
            gpd_args
                Optional arguments to pass to `fit_conditional_marginal_dist`.

        """
        if self.model is None:
            raise GeminiError(
                "Cannot generate scenarios until a model has been fitted!")

        self.model.get_forecast(forecast_df)

        ## use conditional marginals for wind 
        if self.asset_type == 'wind':
            self.model.fit_conditional_marginal_dist(**gpd_args)
        
        if self.meta_df is None:
            upper_dict = None
        else:
            upper_dict = self.meta_df.Capacity

        self.model.generate_gauss_scenarios(nscen, upper_dict=upper_dict)
        self.scenarios[self.asset_type] = self.model.scen_df
        self.forecasts[self.asset_type] = self.get_forecast(forecast_df)

    def get_hist_df_dict(
            self,
            assets: Optional[Iterable[str]] = None
            ) -> Dict[str, pd.DataFrame]:
        """Utility for getting historical values for a given set of assets."""

        if assets is None:
            assets = self.asset_list
        else:
            assets = sorted(assets)

        return {
            'actual': self.hist_actual_df.loc[:, assets],
            'forecast': self.hist_forecast_df.loc[
                        :, ['Issue_time', 'Forecast_time'] + assets]
            }

    def asset_distance(self,
                       assets: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """Utility for calculating distances between asset locations."""

        if self.meta_df is None:
            raise GeminiError(
                "Cannot compute asset distances without metadata!")

        if assets is None:
            assets = self.asset_list

        else:
            for asset in assets:
                if asset not in self.meta_df.index:
                    raise GeminiError("Given asset `{}` does not have a "
                                      "metadata entry!".format(asset))

        dist = pd.DataFrame(0., index=assets, columns=assets)

        # calculate Euclidean distances between all pairs of assets
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                dist_val = np.sqrt(
                    (self.meta_df.latitude[assets[i]]
                     - self.meta_df.latitude[assets[j]]) ** 2
                    + (self.meta_df.longitude[assets[i]]
                       - self.meta_df.longitude[assets[j]]) ** 2
                    )

                dist.loc[assets[i], assets[j]] = dist_val
                dist.loc[assets[j], assets[i]] = dist_val

        return dist

    def get_forecast(self, forecast_df: pd.DataFrame) -> pd.Series:
        """Get forecasts issued for the period scenarios are generated for."""

        use_forecasts = forecast_df[
            forecast_df['Issue_time'] == self.forecast_issue_time].drop(
                columns='Issue_time').set_index('Forecast_time')

        use_forecasts = use_forecasts[
            (use_forecasts.index >= self.scen_start_time)
            & (use_forecasts.index <= self.scen_end_time)
            ].sort_index()
        use_forecasts.index = self.scen_timesteps

        return use_forecasts.unstack()

    def get_yearly_date_range(
            self,
            use_date: pd.Timestamp, num_of_days: Optional[int] = None,
            ) -> Set[pd.Timestamp]:
        """Gets a historical date range around a given day and time for model training.

        Arguments
        ---------
            use_date        The date and time around which the range will be centered.
            num_of_days     The "radius" of the range. If not given, all
                            historical days will be used instead.

        """
        hist_dates = set(pd.date_range(
            start=self.hist_start, end=self.hist_end, freq='D', tz='utc'))

        if num_of_days is not None:

            near_dates = set()
            relative_years = {hist_date.year - use_date.year for hist_date in hist_dates}

            for relative_year in relative_years:
                year_date = use_date + relativedelta(years = relative_year)

                near_dates.update(pd.date_range(
                    start=year_date - pd.Timedelta(num_of_days, unit='D'),
                    periods=2 * num_of_days + 1, freq='D', tz='utc')
                    )

            hist_dates &= near_dates

        return sorted(list(hist_dates))

    def write_to_csv(self,
                     save_dir: Union[str, Path],
                     actual_dfs: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                     write_forecasts: bool = True) -> None:
        """
        This function saves the scenarios generated by this engine to file.

        A .csv is created for each asset, with scenario time points as
        columns and a row for each scenario. A row will also be added for the
        corresponding actual asset values, and optionally, another row for the
        forecasted values for the scenario time points. These files are stored
        in a subdirectory <scen_date>/<asset_type> of `save_dir`, e.g.
        20170811/load/Coast.csv or 20180102/solar/Prickly_Pear.csv.

        Arguments
        ---------
            save_dir
                The directory where output files will be saved.
            actual_dfs
                The actual asset values for the scenario time points.
                If a dataframe is given, it is assumed to have assets of the
                same type as the engine's assets; otherwise, it must be a
                dictionary with asset types as keys and actual values as
                values, e.g. {'load': ..., 'solar': ...}.
            write_forecasts
                Should forecasted asset values be appended to the output file?

        """
        if not isinstance(actual_dfs, dict):
            actual_dfs = {self.asset_type: actual_dfs}

        scen_date = str(self.scen_start_time.strftime('%Y%m%d'))
        for asset_type, forecasts in self.forecasts.items():
            out_dir = Path(save_dir, scen_date, asset_type)

            # create the directory where output files will be stored, as well
            # as the first two columns of the output containing meta-data
            os.makedirs(out_dir, exist_ok=True)
            scen_count = self.scenarios[asset_type].shape[0]
            scen_types = ['Simulation'] * scen_count
            scen_indxs = [i + 1 for i in range(scen_count)]

            if write_forecasts:
                scen_types = ['Forecast'] + scen_types
                scen_indxs = [1] + scen_indxs

            if actual_dfs[asset_type] is not None:
                scen_types = ['Actual'] + scen_types
                scen_indxs = [1] + scen_indxs

            # create the output data table for each generator of this type
            out_index = pd.DataFrame({'Type': scen_types, 'Index': scen_indxs})
            for asset in forecasts.index.unique(0):
                out_vals = self.scenarios[
                    asset_type][asset][self.scen_timesteps].T

                # add the asset forecast values for the scenario time points
                if write_forecasts:
                    out_vals = pd.concat([
                        forecasts[asset][self.scen_timesteps], out_vals],
                        ignore_index=True, axis=1
                        )

                # add the asset actual values for the scenario time points
                if actual_dfs[asset_type] is not None:
                    out_vals = pd.concat([
                        actual_dfs[asset_type].loc[self.scen_timesteps, asset],
                        out_vals
                        ], ignore_index=True, axis=1)

                # concatenate the scenario value columns and the meta-data
                # columns to get the final output table
                out_vals.columns = out_index.index
                out_vals.index = [ts.strftime('%H%M') for ts in out_vals.index]
                out_table = pd.concat([out_index, out_vals.T], axis=1)

                # TODO: round values for more compact storage
                # save output table to file
                filename = asset.rstrip().replace(' ', '_').replace('/', '_') + '.csv'
                out_table.to_csv(out_dir / filename, index=False)
