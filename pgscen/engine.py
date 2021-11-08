"""Engines for generating power grid asset scenarios using historical data."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from operator import itemgetter
from astral import Observer
from astral.sun import sun
from typing import List, Dict, Set, Iterable, Optional, Union

from .model import get_asset_list, GeminiModel, GeminiError
from .utils.solar_utils import overlap, get_yearly_date_range


class GeminiEngine(object):
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

        forecasts : Dict[pd.Series]
            The forecasted values for the scenario time window which were used
            as a basis to generate scenarios.
        scenarios : Dict[pd.DataFrame]
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
        self.meta_df = meta_df
        self.asset_type = asset_type
        self.model = None

        # standardize the format of the meta-information dataframe
        if meta_df is not None:

            # solar case
            if 'site_ids' in meta_df.columns:
                self.meta_df = self.meta_df.sort_values('site_ids').set_index(
                    'site_ids', verify_integrity=True)

            # wind case
            elif 'Facility.Name' in meta_df.columns:
                self.meta_df = self.meta_df.sort_values(
                    'Facility.Name').set_index('Facility.Name').rename(
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

    def fit(self, asset_rho: float, horizon_rho: float) -> None:
        """
        This function creates and fits a scenario model using historical asset
        values. The model will estimate the distributions of the deviations
        from actual values observed in the forecast dataset.

        Arguments
        ---------
            asset_rho
                Hyper-parameter governing the covariances between assets.
            horizon_rho
                Hyper-parameter governing the covariances between time points.

        """
        self.model = GeminiModel(self.scen_start_time, self.get_hist_df_dict(),
                                 None, None,
                                 self.forecast_resolution_in_minute,
                                 self.num_of_horizons,
                                 self.forecast_lead_hours)

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
                Optional arguments to pass to `fit_conditional_gpd`.

        """
        if self.model is None:
            raise GeminiError(
                "Cannot generate scenarios until a model has been fitted!")

        self.model.get_forecast(forecast_df)
        self.model.fit_conditional_gpd(self.asset_type, **gpd_args)
        self.model.generate_gauss_scenarios(nscen)

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

        # get formatted labels for the scenario time points and start time
        fmt_timesteps = [ts.strftime('%H%M') for ts in self.scen_timesteps]
        for asset_type, forecast in self.forecasts.items():
            scen_date = str(self.scen_start_time.strftime('%Y%m%d'))

            # create the directory where output files will be stored
            out_dir = Path(save_dir, scen_date, asset_type)
            if not out_dir.exists():
                os.makedirs(out_dir)

            # TODO: make these concatenations cleaner
            # create the output data table
            for asset in forecast.index.unique(0):
                df = pd.DataFrame(columns=['Type', 'Index'] + fmt_timesteps)

                # add the asset actual values for the scenario time points
                if actual_dfs[asset_type] is not None:
                    actu_arr = np.reshape(
                        actual_dfs[asset_type][asset][
                            (actual_dfs[asset_type][asset].index
                             >= self.scen_start_time)
                            & (actual_dfs[asset_type][asset].index
                               <= self.scen_end_time)
                            ].sort_index().values,
                        (1, self.num_of_horizons)
                        )

                    df = df.append(
                        pd.concat([pd.DataFrame([['Actual', 1]],
                                                columns=['Type', 'Index']),
                                   pd.DataFrame(data=actu_arr,
                                                columns=fmt_timesteps)],
                                  axis=1)
                        )

                # add the asset forecast values for the scenario time points
                if write_forecasts:
                    df = df.append(
                        pd.concat([pd.DataFrame([['Forecast', 1]],
                                                columns=['Type', 'Index']),
                                   pd.DataFrame(data=forecast[asset].values,
                                                index=fmt_timesteps).T],
                                  axis=1)
                        )

                # add the asset scenario values
                scen_count = self.scenarios[asset_type].shape[0]
                df = df.append(
                    pd.concat([
                        pd.DataFrame(
                            data=np.concatenate(([['Simulation']] * scen_count,
                                                 [[i + 1]
                                                  for i in range(scen_count)]),
                                                axis=1),
                            columns=['Type', 'Index']
                            ),
                        pd.DataFrame(
                            data=self.scenarios[asset_type][asset].values,
                            columns=fmt_timesteps
                            )
                        ], axis=1)
                    )

                # TODO: round values for more compact storage
                # save output table to file
                filename = asset.rstrip().replace(' ', '_') + '.csv'
                df.to_csv(out_dir / filename, index=False)


class SolarGeminiEngine(GeminiEngine):
    """
    A class for generating scenarios using photovoltaic asset values.

    Attributes
    ----------
        asset_locs: Dict[str, Observer]
            Asset locations which allow for retrieval of meta information about
            sunsets and sunrises.

        gemini_dict: Dict[Union[str, Tuple[str, int]], Dict]
            Models for each time of day covered by the scenario.
            There will be a 'day' model, as well as conditional models for
            time periods when the sun is coming up or down that must be handled
            separately.

        cond_count: int
            The number of conditional dusk/dawn models.
        asset_distance_mat: pd.DataFrame
            The distances between the locations of each pair of assets.
        time_shift: int
            How many hours these assets' location time zone differs from UTC.
        us_state: str
            Which US state the assets are located in. The values supported
            currently are "Texas" for the default ERCOT/NREL datasets and
            "California" for RTS-GMLC.

    """

    def __init__(self,
                 solar_hist_actual_df: pd.DataFrame,
                 solar_hist_forecast_df: pd.DataFrame,
                 scen_start_time: pd.Timestamp, solar_meta_df: pd.DataFrame,
                 forecast_resolution_in_minute: int = 60,
                 num_of_horizons: int = 24,
                 forecast_lead_time_in_hour: int = 12,
                 us_state: str = 'Texas') -> None:

        super().__init__(solar_hist_actual_df, solar_hist_forecast_df,
                         scen_start_time, solar_meta_df, 'solar',
                         forecast_resolution_in_minute, num_of_horizons,
                         forecast_lead_time_in_hour)

        self.asset_locs = {site: Observer(lat, lon)
                           for site, lat, lon in zip(solar_meta_df.site_ids,
                                                     solar_meta_df.latitude,
                                                     solar_meta_df.longitude)}

        # get sunrise and sunset times for the scenario start time at each of
        # the asset locations
        local_date = self.scen_start_time.tz_convert('US/Central').date()
        asset_suns = {site: sun(loc, date=local_date)
                      for site, loc in self.asset_locs.items()}

        # get the range of sunrise and sunset times at these assets' locations
        first_sunrise, last_sunrise = None, None
        first_sunset, last_sunset = None, None

        for s in asset_suns.values():
            sunrise = pd.to_datetime(s['sunrise'])
            sunset = pd.to_datetime(s['sunset'])

            if first_sunrise is None or sunrise < first_sunrise:
                first_sunrise = sunrise
            if last_sunrise is None or sunrise > last_sunrise:
                last_sunrise = sunrise

            if first_sunset is None or sunset < first_sunset:
                first_sunset = sunset
            if last_sunset is None or sunset > last_sunset:
                last_sunset = sunset

        sunrise_period = (first_sunrise, last_sunrise)
        sunset_period = (first_sunset, last_sunset)
        stepsize = pd.Timedelta(self.forecast_resolution_in_minute, unit='min')
        one_hour = pd.Timedelta(1, unit='H')

        # get the scenario time windows which overlap with sunrise or sunset
        # times for at least some of the assets
        sunrise_prms = [
            {'asset_list': sorted(site for site, s in asset_suns.items()
                              if (pd.to_datetime(s['sunrise'])
                                  < (ts + pd.Timedelta(50, unit='minute')))),
             'scen_start_time': ts,
             'num_of_horizons': 1 + (horizon < len(self.scen_timesteps) - 1),
             'forecast_lead_hours': self.forecast_lead_hours + horizon}
            for horizon, ts in enumerate(self.scen_timesteps)
            if overlap(sunrise_period, (ts, ts + one_hour))
            ]

        sunset_prms = [
            {'asset_list': sorted(site for site, s in asset_suns.items()
                              if (pd.to_datetime(s['sunset'])
                                  > (ts + pd.Timedelta(10, unit='minute')))),
             'scen_start_time': ts - stepsize,
             'num_of_horizons': 1 + (horizon > 0),
             'forecast_lead_hours': (self.forecast_lead_hours + horizon - 1)}
            for horizon, ts in enumerate(self.scen_timesteps)
            if overlap(sunset_period, (ts, ts + one_hour))
            ]

        sunrise_prms = [prms for prms in sunrise_prms if prms['asset_list']]
        sunset_prms = [prms for prms in sunset_prms if prms['asset_list']]

        # TODO: make this cleaner
        # get the daytime scenario windows which do not overlap with any
        # sunsets or sunrises
        day_horizons = [horizon
                        for horizon, ts in enumerate(self.scen_timesteps)
                        if (not overlap(sunrise_period, (ts, ts + one_hour))
                            and not overlap(sunset_period, (ts, ts + one_hour))
                            and last_sunrise < ts < first_sunset)]
        day_lead_hours = self.forecast_lead_hours + day_horizons[0]

        # instantiate the data structure that will store fitted daytime
        # scenario models as well as the dawn/dusk conditional scenario models
        self.gemini_dict = {
            'day': {'asset_list': self.asset_list,
                    'scen_start_time': self.scen_timesteps[day_horizons[0]],
                    'num_of_horizons': len(day_horizons),
                    'forecast_lead_hours': day_lead_hours,
                    'conditional_model': None}
            }

        cond_indx = 0

        # the last sunrise model is conditional on the daytime model
        self.gemini_dict['cond', cond_indx] = {
            'sun': 'rise', 'conditional_model': 'day', **sunrise_prms[-1]}
        cond_indx += 1

        # the remaining dawn models are conditional on the dawn model that
        # comes after it in the morning
        for prms in sunrise_prms[-2::-1]:
            self.gemini_dict['cond', cond_indx] = {
                'sun': 'rise', 'conditional_model': ('cond', cond_indx - 1),
                **prms
                }
            cond_indx += 1

        # the first sunset model is conditional on the daytime model
        self.gemini_dict['cond', cond_indx] = {
            'sun': 'set', 'conditional_model': 'day', **sunset_prms[0]}
        cond_indx += 1

        # the remaining dusk models are conditional on the dusk model that
        # comes before it in the evening
        for prms in sunset_prms[1:]:
            self.gemini_dict['cond', cond_indx] = {
                'sun': 'set', 'conditional_model': ('cond', cond_indx - 1),
                **prms
                }
            cond_indx += 1

        self.cond_count = cond_indx
        self.asset_distance_mat = self.asset_distance()

        # set time shift relative to UTC based on the state the assets are in
        if us_state == 'Texas':
            self.time_shift = 6
        elif us_state == 'California':
            self.time_shift = 8

        else:
            raise ValueError("The only US states currently supported are "
                             "Texas and California!")

        self.us_state = us_state

    def fit_solar_model(self,
                        hist_start: str = '2017-01-01',
                        hist_end: str = '2018-12-31') -> None:
        """Fit each of the solar scenario models in the engine."""

        # for each solar model, starting with the daytime model, find the
        # historical dates with similar sunsets/rises to the scenario date
        for mdl in ['day'] + [('cond', i) for i in range(self.cond_count)]:
            asset_list = self.gemini_dict[mdl]['asset_list']
            hour = self.gemini_dict[mdl]['scen_start_time'].hour

            if mdl == 'day':
                minute_range = 30
            else:
                minute_range = 10

            hist_dates = self.get_solar_hist_dates(
                self.scen_start_time.floor('D'), asset_list,
                hist_start, hist_end, time_range_in_minutes=minute_range
                )

            # shift hours in the historical date due to utc and local time zone
            if hour >= self.time_shift:
                self.gemini_dict[mdl]['hist_deviation_index'] = [
                    date + pd.Timedelta(hour, unit='H')
                    for date in hist_dates
                    ]
            else:
                self.gemini_dict[mdl]['hist_deviation_index'] = [
                    date + pd.Timedelta(24 + hour, unit='H')
                    for date in hist_dates
                    ]

            # create and fit the solar scenario model for this time of day
            solar_md = GeminiModel(
                self.gemini_dict[mdl]['scen_start_time'],
                self.get_hist_df_dict(asset_list), None,
                self.gemini_dict[mdl]['hist_deviation_index'],
                self.forecast_resolution_in_minute,
                self.gemini_dict[mdl]['num_of_horizons'],
                self.gemini_dict[mdl]['forecast_lead_hours']
                )

            solar_md.fit(self.get_solar_reg_param(asset_list), 1e-2)
            self.gemini_dict[mdl]['gemini_model'] = solar_md

    def fit_load_solar_joint_model(self,
                                   load_hist_actual_df: pd.DataFrame,
                                   load_hist_forecast_df: pd.DataFrame,
                                   hist_start: str = '2017-01-01',
                                   hist_end: str = '2018-12-31',
                                   load_zonal: bool = True) -> None:
        """
        This function fits a joint load/solar model for each time of day. The
        historical datasets for bus loads are given in the same format as they
        would be to instantiate a load-only `GeminiEngine`.
        """

        self.load_zonal = load_zonal
        load_zone_list = get_asset_list(load_hist_actual_df,
                                        load_hist_forecast_df)

        ###################### Base Daytime Model #############################

        # determine historical dates which have similar sunset and sunrise
        # times to the scenario date
        day_hist_dates = self.get_solar_hist_dates(
            self.scen_start_time.floor('D'), self.asset_list,
            hist_start, hist_end, time_range_in_minutes=30
            )

        # shift solar historical dates by the hour of scenario start time due
        # to utc and local time zone
        solar_hour = self.gemini_dict['day']['scen_start_time'].hour
        if solar_hour >= self.time_shift:
            self.gemini_dict['day']['solar_hist_deviation_index'] = [
                date + pd.Timedelta(solar_hour, unit='H')
                for date in day_hist_dates
                ]

        else:
            self.gemini_dict['day']['solar_hist_deviation_index'] = [
                date + pd.Timedelta(24 + solar_hour, unit='H')
                for date in day_hist_dates
                ]

        # shift load historical dates by the hour of scenario start time due to
        # utc and local time zone
        load_hour = self.scen_start_time.hour
        load_hist_dates = get_yearly_date_range(
            self.scen_start_time.floor('D'), 60, hist_start, hist_end)

        if load_hour >= self.time_shift:
            self.gemini_dict['day']['load_hist_deviation_index'] = [
                date + pd.Timedelta(load_hour, unit='H')
                for date in load_hist_dates
                ]
        else:
            self.gemini_dict['day']['load_hist_deviation_index'] = [
                date + pd.Timedelta(24 + load_hour, unit='H')
                for date in load_hist_dates
                ]

        # create and fit the load-only model for day time
        load_md = GeminiModel(
            self.scen_start_time,
            {'actual': load_hist_actual_df, 'forecast': load_hist_forecast_df},
            None, self.gemini_dict['day']['load_hist_deviation_index'],
            self.forecast_resolution_in_minute, self.num_of_horizons,
            self.forecast_lead_hours
            )

        load_md.fit(1e-2, 1e-2)
        self.gemini_dict['day']['load_model'] = load_md

        # create and fit the solar-only model for day time
        solar_md = GeminiModel(
            self.gemini_dict['day']['scen_start_time'],
            self.get_hist_df_dict(), None,
            self.gemini_dict['day']['solar_hist_deviation_index'],
            self.forecast_resolution_in_minute,
            self.gemini_dict['day']['num_of_horizons'],
            self.gemini_dict['day']['forecast_lead_hours']
            )

        solar_md.fit(self.get_solar_reg_param(), 5e-2)
        self.gemini_dict['day']['solar_model'] = solar_md

        # get load data for to the same horizons in solar model
        horizon_shift = int(
            (self.gemini_dict['day']['scen_start_time']
             - self.scen_start_time)
            / pd.Timedelta(self.forecast_resolution_in_minute, unit='min')
            )

        load_gauss_df = pd.DataFrame({
            (zone, horizon): load_md.gauss_df[(zone, horizon_shift + horizon)]
            for zone in load_zone_list
            for horizon in range(solar_md.num_of_horizons)
            })

        load_gauss_df.index += horizon_shift * pd.Timedelta(
            self.forecast_resolution_in_minute, unit='min')

        if load_zonal:
            # get zonal solar data, add prefix to differentiate
            # between load and solar zones
            solar_gauss_df = pd.DataFrame({
                ('_'.join(['Solar', zone]), horizon): solar_md.gauss_df[
                    [(site, horizon) for site in sites]].sum(axis=1)
                for zone, sites in self.meta_df.groupby('Zone').groups.items()
                for horizon in range(solar_md.num_of_horizons)
                })

            # standardize zonal data
            solar_zone_gauss_df_mean = solar_gauss_df.mean()
            solar_zone_gauss_df_std = solar_gauss_df.std()
            solar_zone_gauss_df = solar_gauss_df - solar_zone_gauss_df_mean
            solar_zone_gauss_df /= solar_zone_gauss_df_std
            self.solar_zone_mean = solar_zone_gauss_df_mean
            self.solar_zone_std = solar_zone_gauss_df_std

        else:
            solar_gauss_df = pd.DataFrame({
                ('_'.join(['Solar', site]), horizon): solar_md.gauss_df[
                    (site, horizon)]
                for site in self.asset_list
                for horizon in range(solar_md.num_of_horizons)
                })

        # create and fit a joint load-solar model for the day time
        joint_md = GeminiModel(
            self.gemini_dict['day']['scen_start_time'], None,
            load_gauss_df.merge(solar_gauss_df, how='inner',
                                left_index=True, right_index=True),
            None, self.forecast_resolution_in_minute,
            self.gemini_dict['day']['num_of_horizons'],
            self.gemini_dict['day']['forecast_lead_hours']
            )

        joint_md.fit(0.05, 0.05)
        self.gemini_dict['day']['joint_model'] = joint_md

        ################## Conditional Dawn/Dusk Models #######################

        for i in range(self.cond_count):
            asset_list = self.gemini_dict['cond', i]['asset_list']

            # determine historical dates which have similar sunset and sunrise
            # times to the scenario date
            solar_hist_dates = self.get_solar_hist_dates(
                self.scen_start_time.floor('D'), asset_list,
                hist_start, hist_end, time_range_in_minutes=10
                )

            # shift hours in the historical date due to utc and local time zone
            solar_hour = self.gemini_dict['cond', i]['scen_start_time'].hour
            if solar_hour >= self.time_shift:
                self.gemini_dict['cond', i]['hist_deviation_index'] = [
                    date + pd.Timedelta(solar_hour, unit='H')
                    for date in solar_hist_dates
                    ]

            else:
                self.gemini_dict['cond', i]['hist_deviation_index'] = [
                    date + pd.Timedelta(24 + solar_hour, unit='H')
                    for date in solar_hist_dates
                    ]

            # create and fit a solar model for this time of dawn/dusk
            solar_md = GeminiModel(
                self.gemini_dict['cond', i]['scen_start_time'],
                self.get_hist_df_dict(asset_list), None,
                self.gemini_dict['cond', i]['hist_deviation_index'],
                self.forecast_resolution_in_minute,
                self.gemini_dict['cond', i]['num_of_horizons'],
                self.gemini_dict['cond', i]['forecast_lead_hours']
                )

            solar_md.fit(self.get_solar_reg_param(asset_list), 1e-2)
            self.gemini_dict['cond', i]['solar_model'] = solar_md

    def create_solar_scenario(self,
                              nscen: int, forecast_df: pd.DataFrame) -> None:
        """
        This function generates solar-only scenarios using the models fit by
        the engine.

        Arguments
        ---------
            nscen
                The number of scenarios to generate.
            forecast_df
                Forecasted asset values that will be added to the deviations
                generated by the model to produce scenarios.

        """
        solar_scens = pd.DataFrame(
            0., index=list(range(nscen)), columns=pd.MultiIndex.from_tuples(
                [(asset, timestep) for asset in self.asset_list
                 for timestep in self.scen_timesteps]
                )
            )

        # for each time of day, starting with the day, get the corresponding
        # solar model and fit a conditional Gaussian model to its deviations
        for mdl in ['day'] + [('cond', i) for i in range(self.cond_count)]:
            solar_md = self.gemini_dict[mdl]['gemini_model']
            solar_md.get_forecast(forecast_df)
            solar_md.fit_conditional_gpd('solar', positive_actual=True)

            # if this is a conditional model, get the scenario data from the
            # model it's conditional upon for the times where they overlap
            if self.gemini_dict[mdl]['conditional_model']:
                cond_md = self.gemini_dict[mdl]['conditional_model']
                cond_solar_md = self.gemini_dict[cond_md]['gemini_model']

                overlap_timesteps = sorted(set(solar_md.scen_timesteps)
                                           & set(cond_solar_md.scen_timesteps))
                solar_horizons = [solar_md.scen_timesteps.index(t)
                                  for t in overlap_timesteps]
                cond_horizons = [cond_solar_md.scen_timesteps.index(t)
                                 for t in overlap_timesteps]

                cond_scen_df = pd.DataFrame({
                    (asset, slr_hz): cond_solar_md.scen_gauss_df[
                        (asset, cond_hz)]
                    for asset in solar_md.asset_list
                    for cond_hz, slr_hz in zip(cond_horizons, solar_horizons)
                    })

                # get parameter values using this model's fitted parameters and
                # its conditional model's generated scenarios for overlap times
                sqrt_cov, mu = solar_md\
                        .conditional_multivar_normal_partial_time(
                            solar_horizons[0], solar_horizons[-1],
                            cond_scen_df
                            )

                # generate scenarios using this model and the
                # conditionally-derived parameters
                solar_md.generate_gauss_scenarios(
                    nscen, sqrt_cov=sqrt_cov, mu=mu,
                    upper_dict=self.meta_df.AC_capacity_MW
                    )

            else:
                solar_md.generate_gauss_scenarios(
                    nscen, upper_dict=self.meta_df.AC_capacity_MW)

            solar_scens.update(solar_md.scen_df)

        # save the generated scenarios and the forecasted asset values for the
        # same time points
        self.scenarios['solar'] = solar_scens
        self.forecasts['solar'] = self.get_forecast(forecast_df)

    def create_load_solar_joint_scenario(
            self,
            nscen: int, load_forecast_df: pd.DataFrame,
            solar_forecast_df: pd.DataFrame
            ) -> None:
        """
        This function generates load-solar jointly-derived scenarios using the
        models fit by the engine.

        Arguments
        ---------
            nscen
                The number of scenarios to generate.
            load_forecast_df
                Forecasted load asset values that will be added to the
                deviations generated by the model to produce load scenarios.
            solar_forecast_df
                Forecasted solar asset values as above.

        """
        solar_scens = pd.DataFrame(
            0., index=list(range(nscen)), columns=pd.MultiIndex.from_tuples(
                [(asset, timestep) for asset in self.asset_list
                 for timestep in self.scen_timesteps]
                )
            )

        # get the models corresponding to day time and generate day time
        # scenarios using the joint load-solar model
        joint_md = self.gemini_dict['day']['joint_model']
        load_md = self.gemini_dict['day']['load_model']
        solar_md = self.gemini_dict['day']['solar_model']
        joint_md.generate_gauss_scenarios(nscen)

        # separate load and solar joint Gaussian scenarios
        horizon_shift = int(
            (self.gemini_dict['day']['solar_model'].scen_start_time
             - self.gemini_dict['day']['load_model'].scen_start_time)
            / pd.Timedelta(self.forecast_resolution_in_minute, unit='min')
            )

        load_joint_scen_df = pd.DataFrame({
            (zone,
             horizon_shift + horizon): joint_md.scen_gauss_df[(zone, horizon)]
            for zone in joint_md.asset_list
            for horizon in range(joint_md.num_of_horizons)
            if not zone.startswith('Solar_')
            })

        solar_joint_scen_df = joint_md.scen_gauss_df[
            [(zone, horizon) for zone in joint_md.asset_list
             for horizon in range(joint_md.num_of_horizons)
             if zone.startswith('Solar_')]
            ]

        # undo zone normalization done during fitting
        if self.load_zonal:
            solar_joint_scen_df = (solar_joint_scen_df * self.solar_zone_std
                                   + self.solar_zone_mean)

        # generate daytime scenarios for load assets conditional on
        # the joint model
        load_md.get_forecast(load_forecast_df)
        load_md.fit_conditional_gpd('load',
                                    bin_width_ratio=0.1, min_sample_size=400)

        cond_horizon_start = int(
            (solar_md.scen_start_time - load_md.scen_start_time)
            / pd.Timedelta(self.forecast_resolution_in_minute, unit='min')
            )

        cond_horizon_end = cond_horizon_start + joint_md.num_of_horizons - 1
        sqrtcov, mu = load_md.conditional_multivar_normal_partial_time(
            cond_horizon_start, cond_horizon_end, load_joint_scen_df)
        load_md.generate_gauss_scenarios(nscen, sqrt_cov=sqrtcov, mu=mu)

        # generate daytime scenarios for photovoltaic assets
        # conditional on the joint model
        solar_md = self.gemini_dict['day']['solar_model']
        solar_md.get_forecast(solar_forecast_df)
        solar_md.fit_conditional_gpd('solar', positive_actual=True)

        solar_joint_scen_df.columns = pd.MultiIndex.from_tuples(
            [(zone[6:], horizon)
             for zone, horizon in solar_joint_scen_df.columns]
            )

        if self.load_zonal:
            membership = self.meta_df.groupby('Zone').groups

            sqrt_cov, mu = solar_md.conditional_multivar_normal_aggregation(
                solar_joint_scen_df, membership)

        else:
            sqrt_cov, mu = solar_md.conditional_multivar_normal_partial_time(
                0, len(solar_md.scen_timesteps) - 1, solar_joint_scen_df)

        solar_md.generate_gauss_scenarios(
            nscen, sqrt_cov=sqrt_cov, mu=mu,
            upper_dict=self.meta_df.AC_capacity_MW
            )
        solar_scens.update(solar_md.scen_df)

        self.gemini_dict['day']['joint_model'] = joint_md
        self.gemini_dict['day']['load_model'] = load_md
        self.gemini_dict['day']['solar_model'] = solar_md

        # for dusk and dawn times, get the corresponding solar model and fit a
        # conditional Gaussian model to its deviations
        for i in range(self.cond_count):
            solar_md = self.gemini_dict['cond', i]['solar_model']
            solar_md.get_forecast(solar_forecast_df)
            solar_md.fit_conditional_gpd('solar', positive_actual=True)

            # get the scenario data from the model this model is conditional
            # upon for the times where they overlap
            cond_md = self.gemini_dict['cond', i]['conditional_model']
            cond_solar_md = self.gemini_dict[cond_md]['solar_model']

            overlap_timesteps = sorted(set(solar_md.scen_timesteps)
                                       & set(cond_solar_md.scen_timesteps))
            solar_horizons = [solar_md.scen_timesteps.index(t)
                              for t in overlap_timesteps]
            cond_horizons = [cond_solar_md.scen_timesteps.index(t)
                             for t in overlap_timesteps]

            cond_scen_df = pd.DataFrame({
                (asset, slr_hz): cond_solar_md.scen_gauss_df[(asset, cond_hz)]
                for asset in solar_md.asset_list
                for cond_hz, slr_hz in zip(cond_horizons, solar_horizons)
                })

            # get parameter values using this model's fitted parameters and
            # its conditional model's generated scenarios for overlap times
            sqrt_cov, mu = solar_md.conditional_multivar_normal_partial_time(
                solar_horizons[0], solar_horizons[-1], cond_scen_df)

            # generate scenarios using this model and the
            # conditionally-derived parameters
            solar_md.generate_gauss_scenarios(
                nscen, sqrt_cov=sqrt_cov, mu=mu,
                upper_dict=self.meta_df.AC_capacity_MW
                )
            solar_scens.update(solar_md.scen_df)

        # save the generated scenarios and the forecasted asset values for the
        # same time points
        self.scenarios['load'] = self.gemini_dict['day']['load_model'].scen_df
        self.scenarios['solar'] = solar_scens
        self.forecasts['load'] = self.get_forecast(load_forecast_df)
        self.forecasts['solar'] = self.get_forecast(solar_forecast_df)

    def get_solar_hist_dates(
            self,
            date: pd.Timestamp, assets: Iterable[str],
            hist_start: str, hist_end: str, time_range_in_minutes: int = 15
            ) -> Set[pd.Timestamp]:
        """
        This function is a utility for finding historical dates which have
        similar sunset and sunrise times as the given assets at the given date.

        Arguments
        ---------
            date
                The date we are comparing historical dates to.
            assets
                The engine assets at whose location we want to compare sunrise
                and sunset times.

            hist_start, hist_end
                The interval of historical dates we will search over.
            time_range_in_minutes
                Maximum difference in sunset and sunrise times for a historical
                date to be considered as similar.

        """
        hist_dates = get_yearly_date_range(date, start=hist_start,
                                           end=hist_end)
        asset_suns = {asset: sun(self.asset_locs[asset], date=date)
                      for asset in assets}

        for site in asset_suns:
            for sun_time in ['sunrise', 'sunset']:
                asset_suns[site][sun_time] = datetime.combine(
                    datetime.min,
                    pd.to_datetime(asset_suns[site][sun_time]).tz_convert(
                        'US/Central').time()
                    )

        cur_rises = pd.Series({site: s['sunrise']
                               for site, s in asset_suns.items()})
        cur_sets = pd.Series(
            {site: s['sunset'] for site, s in asset_suns.items()})

        hist_suns = {asset: {hist_date: sun(self.asset_locs[asset],
                                            date=hist_date)
                            for hist_date in hist_dates}
                     for asset in assets}

        for site in asset_suns:
            for hist_date in hist_dates:
                for sun_time in ['sunrise', 'sunset']:
                    hist_suns[site][hist_date][sun_time] = datetime.combine(
                        datetime.min,
                        pd.to_datetime(
                            hist_suns[site][hist_date][sun_time]).tz_convert(
                            'US/Central'
                            ).time()
                        )

        sun_df = pd.DataFrame(hist_suns)
        sunrise_df = sun_df.applymap(itemgetter('sunrise'))
        sunset_df = sun_df.applymap(itemgetter('sunset'))

        max_diff = pd.Timedelta(time_range_in_minutes, unit='min')
        rise_diffs = (sunrise_df - cur_rises).abs() <= max_diff
        set_diffs = (sunset_df - cur_sets).abs() <= max_diff
        hist_stats = (rise_diffs & set_diffs).all(axis=1)

        return {hist_time for hist_time, hist_stat in hist_stats.iteritems()
                if hist_stat}

    def get_solar_reg_param(self,
                            assets: Optional[List[str]] = None) -> np.array:
        """
        This function is a utility for calculating a regularization
        hyper-parameter matrix to use for GEMINI models based upon the
        distances between photovoltaic asset locations.

        Arguments
        ---------
            assets
                The assets to use when calculating the regularization value.
                If not given, all of the solar assets loaded by this engine
                are used.

        """
        if assets is None:
            assets = self.asset_list

        rho = self.asset_distance_mat.loc[assets, assets].values

        # normalize distance such that largest entry is equal to 0.1
        if np.max(rho) > 0:
            rho /= np.max(rho) * 10

        # set the distance between asset at the same location to be a small
        # positive constant to prevent glasso from not converging
        if (rho > 0).any():
            rho[rho == 0] = 1e-2 * np.min(rho[rho > 0])
        else:
            rho += 1e-4

        return rho
