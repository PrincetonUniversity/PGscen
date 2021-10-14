
import os
from pathlib import Path
import numpy as np
import pandas as pd
from abc import ABC
from datetime import datetime

from astral import LocationInfo
from .model import get_asset_list, GeminiModel, GeminiError
from .utils.solar_utils import sun, overlap, get_solar_hist_dates


class GeminiEngine(ABC):

    def __init__(self, hist_actual_df, hist_forecast_df, scen_start_time,
                 meta_df=None, asset_type=None,
                 forecast_resolution_in_minute=60, num_of_horizons=24,
                 forecast_lead_time_in_hour=12):

        self.asset_list = get_asset_list(hist_actual_df, hist_forecast_df)
        self.num_of_assets = len(self.asset_list)

        self.hist_actual_df = hist_actual_df
        self.hist_forecast_df = hist_forecast_df
        self.scen_start_time = scen_start_time
        self.meta_df = meta_df
        self.asset_type = asset_type
        self.model = None

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

        self.forecast_resolution_in_minute = forecast_resolution_in_minute
        self.num_of_horizons = num_of_horizons
        self.forecast_lead_hours = forecast_lead_time_in_hour

        self.forecast_issue_time = (
            self.scen_start_time - pd.Timedelta(self.forecast_lead_hours,
                                                unit='H')
            )

        self.scen_end_time = (
                self.scen_start_time
                + pd.Timedelta((self.num_of_horizons - 1)
                               * self.forecast_resolution_in_minute,
                               unit='min')
            )

        self.scen_timesteps = pd.date_range(
            start=self.scen_start_time, end=self.scen_end_time,
            freq=str(self.forecast_resolution_in_minute) + 'min'
            ).strftime('%H%M').tolist()

        self.forecasts = dict()
        self.scenarios = dict()

    def fit(self, asset_rho, horizon_rho):
        self.model = GeminiModel(self.scen_start_time, self.get_hist_df_dict(),
                                 None, None,
                                 self.forecast_resolution_in_minute,
                                 self.num_of_horizons,
                                 self.forecast_lead_hours)

        self.model.fit(asset_rho, horizon_rho)

    def create_scenario(self, nscen, forecast_df, **gpd_args):
        self.model.get_forecast(forecast_df)
        self.model.fit_conditional_gpd(self.asset_type, **gpd_args)
        self.model.generate_gauss_scenarios(nscen)

        self.scenarios[self.asset_type] = self.model.scen_df
        self.forecasts[self.asset_type] = self.get_forecast(forecast_df)

    def get_hist_df_dict(self, assets=None):
        if assets is None:
            assets = self.asset_list
        else:
            assets = sorted(assets)

        return {
            'actual': self.hist_actual_df.loc[:, assets],
            'forecast': self.hist_forecast_df.loc[
                        :, ['Issue_time', 'Forecast_time'] + assets]
            }

    def asset_distance(self, assets=None):
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

    def get_forecast(self, forecast_df):
        use_forecasts = forecast_df[
            forecast_df['Issue_time'] == self.forecast_issue_time].drop(
            columns='Issue_time').set_index('Forecast_time')

        use_forecasts = use_forecasts[
            (use_forecasts.index >= self.scen_start_time)
            & (use_forecasts.index <= self.scen_end_time)
            ].sort_index()
        use_forecasts.index = self.scen_timesteps

        return use_forecasts.unstack()

    def write_to_csv(self, save_dir, actual_dfs, write_forecasts=True):
        if not isinstance(actual_dfs, dict):
            actual_dfs = {self.asset_type: actual_dfs}

        for asset_type, forecast in self.forecasts.items():
            scen_date = str(self.scen_start_time.strftime('%Y%m%d'))
            out_dir = Path(save_dir, scen_date, asset_type)

            if not out_dir.exists():
                os.makedirs(out_dir)

            # TODO: make these concatenations cleaner
            for asset in forecast.index.unique(0):
                df = pd.DataFrame(
                    columns=['Type', 'Index'] + self.scen_timesteps)

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
                                                columns=self.scen_timesteps)],
                                  axis=1)
                        )

                if write_forecasts:
                    df = df.append(
                        pd.concat([pd.DataFrame([['Forecast', 1]],
                                                columns=['Type', 'Index']),
                                   pd.DataFrame(forecast[asset]).T],
                                  axis=1)
                        )

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
                        self.scenarios[asset_type][asset]
                        ], axis=1)
                    )

                # TODO: round values for more compact storage
                filename = asset.rstrip().replace(' ', '_') + '.csv'
                df.to_csv(out_dir / filename, index=False)


class SolarGeminiEngine(GeminiEngine):

    def __init__(self, solar_hist_actual_df, solar_hist_forecast_df,
                 scen_start_time, solar_meta_df,
                 forecast_resolution_in_minute=60, num_of_horizons=24,
                 forecast_lead_time_in_hour=12):
        super().__init__(solar_hist_actual_df, solar_hist_forecast_df,
                         scen_start_time, solar_meta_df, 'solar',
                         forecast_resolution_in_minute, num_of_horizons,
                         forecast_lead_time_in_hour)

        stepsize = pd.Timedelta(self.forecast_resolution_in_minute, unit='min')
        scen_timesteps = pd.date_range(start=self.scen_start_time,
                                       periods=self.num_of_horizons,
                                       freq=stepsize)

        # Get scenario date
        local_date = self.scen_start_time.tz_convert('US/Central').date()
        asset_suns = {
            site: sun(LocationInfo(site, 'Texas', 'USA', lat, lon).observer,
                      date=local_date)
            for site, lat, lon in zip(self.meta_df.index,
                                      self.meta_df.latitude,
                                      self.meta_df.longitude)
            }

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

        print('sun rise set times:')
        print(first_sunrise, last_sunrise, first_sunset, last_sunset)

        # Determine model parameters
        sunrise_period = (first_sunrise, last_sunrise)
        sunset_period = (first_sunset, last_sunset)
        one_hour = pd.Timedelta(1, unit='H')

        sunrise_prms = [
            {'asset_list': sorted(site for site, s in asset_suns.items()
                              if (pd.to_datetime(s['sunrise'])
                                  < (ts + pd.Timedelta(50, unit='minute')))),
             'scenario_start_time': ts,
             'num_of_horizons': 1 + (horizon < len(scen_timesteps) - 1),
             'forecast_lead_hours': self.forecast_lead_hours + horizon}
            for horizon, ts in enumerate(scen_timesteps)
            if overlap(sunrise_period, (ts, ts + one_hour))
            ]

        sunset_prms = [
            {'asset_list': sorted(site for site, s in asset_suns.items()
                              if (pd.to_datetime(s['sunset'])
                                  > (ts + pd.Timedelta(10, unit='minute')))),
             'scenario_start_time': ts - stepsize,
             'num_of_horizons': 1 + (horizon > 0),
             'forecast_lead_hours': (self.forecast_lead_hours + horizon - 1)}
            for horizon, ts in enumerate(scen_timesteps)
            if overlap(sunset_period, (ts, ts + one_hour))
            ]

        sunrise_prms = [prms for prms in sunrise_prms if prms['asset_list']]
        sunset_prms = [prms for prms in sunset_prms if prms['asset_list']]

        # TODO: make this cleaner
        day_horizons = [horizon for horizon, ts in enumerate(scen_timesteps)
                        if (not overlap(sunrise_period, (ts, ts + one_hour))
                            and not overlap(sunset_period, (ts, ts + one_hour))
                            and last_sunrise < ts < first_sunset)]
        day_lead_hours = self.forecast_lead_hours + day_horizons[0]

        ################# Determine conditional models ########################
        self.gemini_dict = {
            'day': {'asset_list': self.asset_list,
                    'scenario_start_time': scen_timesteps[day_horizons[0]],
                    'num_of_horizons': len(day_horizons),
                    'forecast_lead_hours': day_lead_hours,
                    'conditional_model': None}
            }

        cond_indx = 0

        self.gemini_dict['cond', cond_indx] = {
            'sun': 'rise', 'conditional_model': 'day', **sunrise_prms[-1]}
        cond_indx += 1

        for prms in sunrise_prms[-2::-1]:
            self.gemini_dict['cond', cond_indx] = {
                'sun': 'rise', 'conditional_model': ('cond', cond_indx - 1),
                **prms
                }
            cond_indx += 1

        self.gemini_dict['cond', cond_indx] = {
            'sun': 'set', 'conditional_model': 'day', **sunset_prms[0]}
        cond_indx += 1

        for prms in sunset_prms[1:]:
            self.gemini_dict['cond', cond_indx] = {
                'sun': 'set', 'conditional_model': ('cond', cond_indx - 1),
                **prms
                }
            cond_indx += 1

        self.cond_count = cond_indx
        self.asset_distance_mat = self.asset_distance()

    def fit_solar_model(self, hist_start='2017-01-01',hist_end='2018-12-31'):
        """Fit solar models with chosen parameters"""

        for mdl in ['day'] + [('cond', i) for i in range(self.cond_count)]:
            asset_list = self.gemini_dict[mdl]['asset_list']
            hour = self.gemini_dict[mdl]['scenario_start_time'].hour

            if mdl == 'day':
                minute_range = 30
            else:
                minute_range = 10

            # Determine historical dates
            hist_dates = get_solar_hist_dates(
                self.scen_start_time.floor('D'), self.meta_df.loc[asset_list],
                hist_start, hist_end, time_range_in_minutes=minute_range
                )

            # Shift hours in the historical date due to utc
            if hour >= 6:
                self.gemini_dict[mdl]['hist_deviation_index'] = [
                    date + pd.Timedelta(hour, unit='H')
                    for date in hist_dates
                    ]
            else:
                self.gemini_dict[mdl]['hist_deviation_index'] = [
                    date + pd.Timedelta(24 + hour, unit='H')
                    for date in hist_dates
                    ]

            solar_md = GeminiModel(
                self.gemini_dict[mdl]['scenario_start_time'],
                self.get_hist_df_dict(asset_list), None,
                self.gemini_dict[mdl]['hist_deviation_index'],
                self.forecast_resolution_in_minute,
                self.gemini_dict[mdl]['num_of_horizons'],
                self.gemini_dict[mdl]['forecast_lead_hours']
                )

            solar_md.fit(self.get_solar_reg_param(asset_list), 1e-2)
            self.gemini_dict[mdl]['gemini_model'] = solar_md

    def fit_load_solar_joint_model(self,
                                   load_hist_actual_df, load_hist_forecast_df,
                                   hist_start='2017-01-01',
                                   hist_end='2018-12-31'):
        """
        Fit load and solar models with chosen parameters:
        The base model for load and solar is joint model.

        """

        load_zone_list = get_asset_list(load_hist_actual_df,
                                        load_hist_forecast_df)

        ####################### Base model ##################################

        # Determine historical dates
        day_hist_dates = get_solar_hist_dates(
            self.scen_start_time.floor('D'), self.meta_df,
            hist_start, hist_end, time_range_in_minutes=30
            )

        # Shift solar historical dates by the hour of scenario start time due to utc
        solar_hour = self.gemini_dict['day']['scenario_start_time'].hour
        if solar_hour >= 6:
            self.gemini_dict['day']['solar_hist_deviation_index'] = [
                date + pd.Timedelta(solar_hour, unit='H')
                for date in day_hist_dates
                ]

        else:
            self.gemini_dict['day']['solar_hist_deviation_index'] = [
                date + pd.Timedelta(24 + solar_hour, unit='H')
                for date in day_hist_dates
                ]

        # Shift load historical dates by the hour of scenario start time due to utc
        load_hour = self.scen_start_time.hour
        load_hist_dates = self.get_yearly_date_range(60, hist_start, hist_end)

        if load_hour >= 6:
            self.gemini_dict['day']['load_hist_deviation_index'] = [
                date + pd.Timedelta(load_hour, unit='H')
                for date in load_hist_dates
                ]
        else:
            self.gemini_dict['day']['load_hist_deviation_index'] = [
                date + pd.Timedelta(24 + load_hour, unit='H')
                for date in load_hist_dates
                ]

        load_md = GeminiModel(
            self.scen_start_time,
            {'actual': load_hist_actual_df, 'forecast': load_hist_forecast_df},
            None, self.gemini_dict['day']['load_hist_deviation_index'],
            self.forecast_resolution_in_minute, self.num_of_horizons,
            self.forecast_lead_hours
            )

        load_md.fit(1e-2, 1e-2)
        self.gemini_dict['day']['load_model'] = load_md

        solar_md = GeminiModel(
            self.gemini_dict['day']['scenario_start_time'],
            self.get_hist_df_dict(), None,
            self.gemini_dict['day']['solar_hist_deviation_index'],
            self.forecast_resolution_in_minute,
            self.gemini_dict['day']['num_of_horizons'],
            self.gemini_dict['day']['forecast_lead_hours']
            )

        solar_md.fit(self.get_solar_reg_param(), 5e-2)
        self.gemini_dict['day']['solar_model'] = solar_md

        # Get load data for to the same horizons in solar model
        horizon_shift = int(
            (self.gemini_dict['day']['scenario_start_time']
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

        # get zonal solar data, add prefix to differentiate
        # between load and solar zones
        solar_zone_gauss_df = pd.DataFrame({
            ('_'.join(['Solar', zone]), horizon): solar_md.gauss_df[
                [(site, horizon) for site in sites]].sum(axis=1)
            for zone, sites in self.meta_df.groupby('Zone').groups.items()
            for horizon in range(solar_md.num_of_horizons)
            })
        solar_zone_list = solar_zone_gauss_df.columns.levels[0].tolist()

        # Standardize zonal data
        solar_zone_gauss_df_mean = solar_zone_gauss_df.mean()
        solar_zone_gauss_df_std = solar_zone_gauss_df.std()
        solar_zone_gauss_df = solar_zone_gauss_df - solar_zone_gauss_df_mean
        solar_zone_gauss_df /= solar_zone_gauss_df_std

        joint_md = GeminiModel(
            self.gemini_dict['day']['scenario_start_time'],
            None, load_gauss_df.merge(solar_zone_gauss_df,
                                      how='inner', left_index=True,
                                      right_index=True),
            None, self.forecast_resolution_in_minute,
            self.gemini_dict['day']['num_of_horizons'],
            self.gemini_dict['day']['forecast_lead_hours']
            )

        joint_md.fit(0.05, 0.05)
        self.gemini_dict['day']['joint_model'] = joint_md
        self.solar_zone_mean = solar_zone_gauss_df_mean
        self.solar_zone_std = solar_zone_gauss_df_std

        ################### Conditional models ##############################

        for i in range(self.cond_count):
            asset_list = self.gemini_dict['cond', i]['asset_list']

            # Determine historical dates
            solar_hist_dates = get_solar_hist_dates(
                self.scen_start_time.floor('D'), self.meta_df.loc[asset_list],
                hist_start, hist_end, time_range_in_minutes=10
                )

            # Shift hours in the historical date due to utc
            solar_hour = self.gemini_dict['cond', i][
                'scenario_start_time'].hour

            if solar_hour >= 6:
                self.gemini_dict['cond', i]['hist_deviation_index'] = [
                    date + pd.Timedelta(solar_hour, unit='H')
                    for date in solar_hist_dates
                    ]
            else:
                self.gemini_dict['cond', i]['hist_deviation_index'] = [
                    date + pd.Timedelta(24 + solar_hour, unit='H')
                    for date in solar_hist_dates
                    ]

            solar_md = GeminiModel(
                self.gemini_dict['cond', i]['scenario_start_time'],
                self.get_hist_df_dict(asset_list), None,
                self.gemini_dict['cond', i]['hist_deviation_index'],
                self.forecast_resolution_in_minute,
                self.gemini_dict['cond', i]['num_of_horizons'],
                self.gemini_dict['cond', i]['forecast_lead_hours']
                )

            solar_md.fit(self.get_solar_reg_param(asset_list), 1e-2)
            self.gemini_dict['cond', i]['solar_model'] = solar_md

    def create_solar_scenario(self, nscen, forecast_df):
        solar_scens = pd.DataFrame(
            0., index=list(range(nscen)), columns=pd.MultiIndex.from_tuples(
                [(asset, timestep) for asset in self.asset_list
                 for timestep in self.scen_timesteps]
                )
            )

        for mdl in ['day'] + [('cond', i) for i in range(self.cond_count)]:
            solar_md = self.gemini_dict[mdl]['gemini_model']
            solar_md.get_forecast(forecast_df)
            solar_md.fit_conditional_gpd('solar', positive_actual=True)

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

                sqrt_cov, mu = solar_md.conditional_multivar_normal_partial_time(
                    solar_horizons[0], solar_horizons[-1], cond_scen_df)

                solar_md.generate_gauss_scenarios(
                    nscen, sqrt_cov=sqrt_cov, mu=mu,
                    upper_dict=self.meta_df.AC_capacity_MW
                    )

            else:
                solar_md.generate_gauss_scenarios(
                    nscen, upper_dict=self.meta_df.AC_capacity_MW)

            solar_scens.update(solar_md.scen_df)

        self.scenarios['solar'] = solar_scens
        self.forecasts['solar'] = self.get_forecast(forecast_df)

    def create_load_solar_joint_scenario(self, nscen,
                                         load_forecast_df, solar_forecast_df):
        """
        Create scenario
        """

        solar_scens = pd.DataFrame(
            0., index=list(range(nscen)), columns=pd.MultiIndex.from_tuples(
                [(asset, timestep) for asset in self.asset_list
                 for timestep in self.scen_timesteps]
                )
            )

        joint_md = self.gemini_dict['day']['joint_model']
        load_md = self.gemini_dict['day']['load_model']
        solar_md = self.gemini_dict['day']['solar_model']
        joint_md.generate_gauss_scenarios(nscen)

        # Separate load and solar Gaussian scenario
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

        solar_joint_scen_df = (solar_joint_scen_df * self.solar_zone_std
                               + self.solar_zone_mean)

        # Generate conditional scenario for load
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

        # Generate conditional scenario for solar
        membership = self.meta_df.groupby('Zone').groups
        solar_md = self.gemini_dict['day']['solar_model']
        solar_md.get_forecast(solar_forecast_df)
        solar_md.fit_conditional_gpd('solar', positive_actual=True)

        solar_joint_scen_df.columns = pd.MultiIndex.from_tuples(
            [(zone[6:], horizon)
             for zone, horizon in solar_joint_scen_df.columns]
            )

        sqrt_cov, mu = solar_md.conditional_multivar_normal_aggregation(
            solar_joint_scen_df, membership)

        solar_md.generate_gauss_scenarios(
            nscen, sqrt_cov=sqrt_cov, mu=mu,
            upper_dict=self.meta_df.AC_capacity_MW
            )
        solar_scens.update(solar_md.scen_df)

        self.gemini_dict['day']['joint_model'] = joint_md
        self.gemini_dict['day']['load_model'] = load_md
        self.gemini_dict['day']['solar_model'] = solar_md

        for i in range(self.cond_count):
            solar_md = self.gemini_dict['cond', i]['solar_model']
            solar_md.get_forecast(solar_forecast_df)
            solar_md.fit_conditional_gpd('solar', positive_actual=True)

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

            sqrt_cov, mu = solar_md.conditional_multivar_normal_partial_time(
                solar_horizons[0], solar_horizons[-1], cond_scen_df)

            solar_md.generate_gauss_scenarios(
                nscen, sqrt_cov=sqrt_cov, mu=mu,
                upper_dict=self.meta_df.AC_capacity_MW
                )
            solar_scens.update(solar_md.scen_df)

        self.scenarios['load'] = self.gemini_dict['day']['load_model'].scen_df
        self.scenarios['solar'] = solar_scens
        self.forecasts['load'] = self.get_forecast(load_forecast_df)
        self.forecasts['solar'] = self.get_forecast(solar_forecast_df)

    def get_yearly_date_range(self, num_of_days=60,
                              start='2017-01-01', end='2018-12-31'):
        """
        Get date range around a specific date
        """
        date = self.scen_start_time.floor('D')
        hist_dates = pd.date_range(start=start,end=end,freq='D',tz='utc')
        hist_years = hist_dates.year.unique()
        hist_dates = set(hist_dates)

        # Take 60 days before and after
        near_dates = set()
        for year in hist_years:
            year_date = datetime(year,date.month,date.day)
            near_dates = near_dates.union(set(pd.date_range(
                start=year_date-pd.Timedelta(num_of_days, unit='D'),
                periods=2*num_of_days+1,freq='D', tz='utc')
                ))

        return hist_dates.intersection(near_dates)

    def get_solar_reg_param(self, assets=None):
        if assets is None:
            assets = self.asset_list

        rho = self.asset_distance_mat.loc[assets, assets].values

        # Normalize distance such that largest entry = 0.1.
        if np.max(rho) > 0:
            rho /= np.max(rho) * 10

        # Set the distance between asset at the same location to be a small
        # positive constant to prevent glasso from not converging
        if (rho > 0).any():
            rho[rho == 0] = 1e-2 * np.min(rho[rho > 0])
        else:
            rho += 1e-4

        return rho
