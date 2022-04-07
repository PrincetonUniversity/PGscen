
import numpy as np
import pandas as pd

from scipy.linalg import sqrtm
from scipy.stats import norm
from sklearn.decomposition import PCA
from astral import LocationInfo
from astral.sun import sun
from typing import List, Dict, Tuple, Iterable, Optional

from .model import GeminiModel
from .engine import GeminiEngine
from .utils.r_utils import qdist, graphical_lasso, gemini, ecdf, standardize
from .utils.solar_utils import (get_yearly_date_range,
                                get_asset_transition_hour_info)


class PCAGeminiEngine(GeminiEngine):

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

        ############# Compute transitional hour delay time ####################

        print('computing hour delay time....')

        hist_dates = solar_hist_forecast_df.groupby(
            'Issue_time').head(1).Forecast_time.dt.date.tolist()

        delay_dict = dict()
        hist_sun_dict = dict()
        sun_fields = ['Time', 'Horizon',
                      'Actual', 'Forecast', 'Deviation', 'Active Minutes']

        asset_locs = {
            asset: LocationInfo(asset, 'Texas', 'USA',
                                asset_data.latitude, asset_data.longitude)
            for asset, asset_data in self.meta_df.iterrows()
            }

        for asset, actl_vals in solar_hist_actual_df.iteritems():
            fcst_vals = solar_hist_forecast_df.set_index(
                'Forecast_time')[asset]

            sunrise_data, sunset_data = list(), list()
            day_data = {'Time': list(), 'Actual': list(),
                        'Forecast': list(), 'Deviation': list()}

            for hist_date in hist_dates:
                trans_info = get_asset_transition_hour_info(asset_locs[asset],
                                                            hist_date)

                # sunrise
                sunrise_actl = actl_vals[trans_info['sunrise']['timestep']]
                sunrise_fcst = fcst_vals[trans_info['sunrise']['timestep']]

                sunrise_data.append([trans_info['sunrise']['time'],
                                     trans_info['sunrise']['timestep'],
                                     sunrise_actl, sunrise_fcst,
                                     sunrise_actl - sunrise_fcst,
                                     trans_info['sunrise']['active']])

                # sunset
                sunset_actl = actl_vals[trans_info['sunset']['timestep']]
                sunset_fcst = fcst_vals[trans_info['sunset']['timestep']]

                sunset_data.append([trans_info['sunset']['time'],
                                    trans_info['sunset']['timestep'],
                                    sunset_actl, sunset_fcst,
                                    sunset_actl - sunset_fcst,
                                    trans_info['sunset']['active']])

                # daytime
                day_actls = actl_vals[
                    (actl_vals.index > trans_info['sunrise']['timestep'])
                    & (actl_vals.index < trans_info['sunset']['timestep'])
                    ].sort_index()

                day_fcsts = fcst_vals[
                    (fcst_vals.index > trans_info['sunrise']['timestep'])
                    & (fcst_vals.index < trans_info['sunset']['timestep'])
                    ].sort_index()

                day_data['Time'] += day_actls.index.tolist()
                day_data['Actual'] += day_actls.values.tolist()
                day_data['Forecast'] += day_fcsts.values.tolist()
                day_data['Deviation'] += list(day_actls - day_fcsts)

            sunrise_df = pd.DataFrame(data=sunrise_data, columns=sun_fields)
            sunset_df = pd.DataFrame(data=sunset_data, columns=sun_fields)
            day_df = pd.DataFrame(day_data).set_index('Time')

            # figure out delay times
            delay_dict[asset] = {
                'sunrise': sunrise_df.groupby(
                    'Active Minutes')['Actual'].any().idxmax() - 1,
                'sunset': sunset_df.groupby(
                    'Active Minutes')['Actual'].any().idxmax() - 1
                }

            hist_sun_dict[asset] = {'sunrise': sunrise_df, 'sunset': sunset_df,
                                    'day': day_df}

        ################ Compute transitional hour statistics #################

        self.trans_delay = delay_dict
        self.hist_sun_info = hist_sun_dict
        self.end_day = self.hist_forecast_df[
            'Forecast_time'].max().strftime('%Y-%m-%d')

        asset_horizons = {
            asset: get_asset_transition_hour_info(
                loc, self.scen_start_time.floor('D'),
                delay_dict[asset]['sunrise'], delay_dict[asset]['sunset']
                )
            for asset, loc in asset_locs.items()
            }

        self.trans_horizon = {
            'sunrise': {asset: horizons['sunrise']
                        for asset, horizons in asset_horizons.items()},
            'sunset': {asset: horizons['sunset']
                       for asset, horizons in asset_horizons.items()}
            }

        # set time shift relative to UTC based on the state the assets are in
        if us_state == 'Texas':
            self.time_shift = 6
        elif us_state == 'California':
            self.time_shift = 8

        else:
            raise ValueError("The only US states currently supported are "
                             "Texas and California!")

        self.us_state = us_state

    def fit(self,
            asset_rho: float, horizon_rho: float,
            num_of_components: int, nearest_days: int = 50) -> None:

        # need to localize historical data?
        days = get_yearly_date_range(self.scen_start_time, end=self.end_day,
                                     num_of_days=nearest_days)

        self.model = PCAGeminiModel(
            self.scen_start_time, self.get_hist_df_dict(), None,
            [d + pd.Timedelta(self.time_shift, unit='H') for d in days],
            self.forecast_resolution_in_minute,
            self.num_of_horizons, self.forecast_lead_hours
            )

        self.model.pca_transform(num_of_components)
        self.model.fit(asset_rho, horizon_rho)

    def create_scenario(self,
                        nscen: int, forecast_df: pd.DataFrame) -> None:

        self.model.get_forecast(forecast_df)
        self.model.generate_gauss_pca_scenarios(
            self.trans_horizon, self.hist_sun_info,
            nscen, upper_dict=self.meta_df.Capacity
            )

        self.scenarios[self.asset_type] = self.model.scen_df
        self.forecasts[self.asset_type] = self.get_forecast(forecast_df)

    def fit_load_solar_joint_model(
            self,
            load_hist_actual_df: pd.DataFrame,
            load_hist_forecast_df: pd.DataFrame,
            asset_rho: float, horizon_rho: float,
            num_of_components: int, nearest_days : int = 50
            ) -> None:

        # Need to localize historical data?
        days = get_yearly_date_range(self.scen_start_time, end=self.end_day,
                                     num_of_days=nearest_days)

        # Fit solar asset-level model
        solar_md = PCAGeminiModel(
            self.scen_start_time, self.get_hist_df_dict(), None,
            [d + pd.Timedelta(self.time_shift, unit='H') for d in days],
            self.forecast_resolution_in_minute, self.num_of_horizons,
            self.forecast_lead_hours
            )

        # solar_md.gauss_mean, solar_md.gauss_std, solar_md.gauss_df = standardize(solar_md.gauss_df, ignore_pointmass=True)
        self.solar_md = solar_md
        solar_md.pca_transform(num_of_components=num_of_components)
        solar_md.fit(asset_rho, horizon_rho)

        self.solar_md = solar_md
        dev_index = solar_md.hist_dev_df.index

        # Fit load model to get deviations
        load_md = GeminiModel(self.scen_start_time,
                                 {'actual': load_hist_actual_df, 'forecast': load_hist_forecast_df},
                                 None, dev_index,
                                 self.forecast_resolution_in_minute,
                                 self.num_of_horizons,
                                 self.forecast_lead_hours)
        # load_md.gauss_mean, load_md.gauss_std, load_md.gauss_df = standardize(load_md.gauss_df)
        load_md.fit(1e-2, 1e-2)
        self.load_md = load_md

        # Get Gaussian data for the joint model

        # Determine zonal solar active horizons
        west = self.meta_df.sort_values(['longitude', 'latitude'], ascending=[True, True]).iloc[0,]
        asset, lon, lat = west._name, west['longitude'], west['latitude']
        joint_model_start = pd.to_datetime(max([sun(LocationInfo('west', 'Texas', 'USA', lat, lon).observer,
            date=dt)['sunrise'] for dt in dev_index])) + pd.Timedelta(60+self.trans_delay[asset]['sunrise'], unit='m')

        east = self.meta_df.sort_values(['longitude', 'latitude'], ascending=[False, True]).iloc[0,]
        asset, lon, lat = east._name, east['longitude'], east['latitude']
        joint_model_end = pd.to_datetime(min([sun(LocationInfo('east', 'Texas', 'USA', lat, lon).observer,
            date=dt)['sunset'] for dt in dev_index])) - pd.Timedelta(60+self.trans_delay[asset]['sunset'], unit='m')

        joint_model_start_hour, joint_model_end_hour = joint_model_start.floor('H').hour, joint_model_end.floor('H').hour
        joint_model_start_timestep = [ts for ts in self.scen_timesteps if ts.hour==joint_model_start_hour][0]
        joint_model_end_timestep = [ts for ts in self.scen_timesteps if ts.hour==joint_model_end_hour][0]
        joint_model_horizon_start = self.scen_timesteps.index(joint_model_start_timestep)
        joint_model_horizon_end = self.scen_timesteps.index(joint_model_end_timestep)

        # Zonal load
        joint_load_df = load_md.gauss_df[[(asset, i) for asset in self.load_md.asset_list
            for i in range(joint_model_horizon_start, joint_model_horizon_end+1)]]

        # Aggreate solar to zonal-level
        solar_zone_gauss_df = pd.DataFrame({
                ('_'.join(['Solar', zone]), horizon): solar_md.gauss_df[
                    [(site, horizon) for site in sites]].sum(axis=1)
                for zone, sites in self.meta_df.groupby('Zone').groups.items()
                for horizon in range(joint_model_horizon_start, joint_model_horizon_end+1)
                })

        # Fit load and solar joint model
        joint_md_forecast_lead_hours = self.forecast_lead_hours + \
                int((joint_model_start_timestep - self.scen_start_time) / \
                pd.Timedelta(self.forecast_resolution_in_minute, unit='min'))
        joint_md = GeminiModel(self.scen_start_time,
                                None,
                                joint_load_df.merge(solar_zone_gauss_df, how='inner',
                                    left_index=True, right_index=True),
                                None,
                                self.forecast_resolution_in_minute,
                                joint_model_horizon_end-joint_model_horizon_start+1,
                                joint_md_forecast_lead_hours)
        joint_md.fit(1e-2, 1e-2)

        self.joint_md = joint_md

    def create_load_solar_joint_scenario(self,
                                         nscen,
                                         load_forecast_df, solar_forecast_df):

        solar_md = self.solar_md
        load_md = self.load_md
        joint_md = self.joint_md

        # Generate joint scenarios
        joint_md.generate_gauss_scenarios(nscen)

        horizon_shift = joint_md.forecast_lead_hours - self.forecast_lead_hours

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

        # generate daytime scenarios for load assets conditional on
        # the joint model
        load_md.get_forecast(load_forecast_df)
        load_md.fit_conditional_gpd('load',
                                    bin_width_ratio=0.1, min_sample_size=400)

        cond_horizon_start = horizon_shift

        cond_horizon_end = cond_horizon_start + joint_md.num_of_horizons - 1
        sqrtcov, mu = load_md.conditional_multivar_normal_partial_time(
            cond_horizon_start, cond_horizon_end, load_joint_scen_df)
        load_md.generate_gauss_scenarios(nscen, sqrt_cov=sqrtcov, mu=mu)

        # generate site-level solar scenarios
        solar_md.get_forecast(solar_forecast_df)

        # (ported conditional_multivar...)
        membership = self.meta_df.groupby('Zone').groups
        aggregates_list = [zone[6:] for zone in solar_joint_scen_df.columns.unique(0).tolist()]
        s_mat = np.zeros((len(aggregates_list), len(solar_md.asset_list)))

        for aggregate, assets in membership.items():
            agg_indx = aggregates_list.index(aggregate)

            for asset in assets:
                s_mat[agg_indx, solar_md.asset_list.index(asset)] = 1.

        U = np.kron(s_mat,
                    np.eye(self.num_of_horizons)[horizon_shift:horizon_shift+joint_md.num_of_horizons, :] @
                    solar_md.pca.components_.T) @ np.diag(solar_md.pca_gauss_std.values)
        b = (solar_joint_scen_df.values - U @ solar_md.pca_gauss_mean).T
        sigma = np.kron(solar_md.asset_cov, solar_md.horizon_cov)
        C = sigma @ U.T @ np.linalg.inv(U @ sigma @ U.T)
        A = np.eye(solar_md.num_of_components * solar_md.num_of_assets) - C @ U
        sqrtcov = sqrtm(A @ sigma @ A.T).real
        mu = C @ b

        solar_md.generate_gauss_pca_scenarios(self.trans_horizon,
                self.hist_sun_info,
                nscen,
                sqrtcov=sqrtcov,
                mu=mu,
                upper_dict=self.meta_df.Capacity
                )

        # save the generated scenarios and the forecasted asset values for the
        # same time points
        self.scenarios['load'] = load_md.scen_df
        self.scenarios['solar'] = solar_md.scen_df
        self.forecasts['load'] = self.get_forecast(load_forecast_df)
        self.forecasts['solar'] = self.get_forecast(solar_forecast_df)


class PCAGeminiModel(GeminiModel):

    def __init__(self,
                 scen_start_time: pd.Timestamp,
                 hist_dfs: Optional[Dict[str, pd.DataFrame]] = None,
                 gauss_df: Optional[pd.DataFrame] = None,
                 dev_index: Optional[Iterable[pd.Timestamp]] = None,
                 forecast_resolution_in_minute: int = 60,
                 num_of_horizons: int = 24,
                 forecast_lead_time_in_hour: int = 12) -> None:

        super().__init__(scen_start_time,
                    hist_dfs,
                    gauss_df,
                    dev_index,
                    forecast_resolution_in_minute,
                    num_of_horizons,
                    forecast_lead_time_in_hour)

    def fit_conditional_marginal_dist(self, trans_timestep, hist_dict, actmin_width: int = 5,
                            fcst_width_ratio: float = 0.05) -> None:
        """
        Fit marginal distribution conditional on the forecast or active minutes
        """

        self.marginal_dict = {}
        for asset in self.asset_list:

            # capacity = meta_df.loc[asset].AC_capacity_MW

            sunrise_hrz, sunrise_active = trans_timestep['sunrise'][asset]['timestep'], trans_timestep['sunrise'][asset]['active']
            sunset_hrz, sunset_active = trans_timestep['sunset'][asset]['timestep'], trans_timestep['sunset'][asset]['active']

            sunrise_df, sunset_df, day_df = hist_dict[asset]['sunrise'], \
                hist_dict[asset]['sunset'], hist_dict[asset]['day']

            for timestep in self.scen_timesteps:
                fcst = self.forecasts[asset, timestep]

                if timestep == sunrise_hrz:
                    # Sunrise horizon
                    lower = max(0, sunrise_active-actmin_width)
                    upper = min(60, sunrise_active+actmin_width)

                    selected_df = sunrise_df[(sunrise_df['Active Minutes'] >= lower)
                                            & (sunrise_df['Active Minutes']<= upper)]

                elif timestep == sunset_hrz:
                    # Sunset horizon
                    lower = max(0, sunset_active-actmin_width)
                    upper = min(60, sunset_active+actmin_width)

                    selected_df = sunset_df[(sunset_df['Active Minutes'] >= lower)
                                            & (sunset_df['Active Minutes']<= upper)]

                elif sunrise_hrz < timestep < sunset_hrz:
                    # Daytime horizons

                    fcst_min, fcst_max = day_df['Forecast'].min(), day_df['Forecast'].max()

                    lower = max(fcst_min, fcst - fcst_width_ratio * (
                                fcst_max - fcst_min))
                    upper = min(fcst_max, fcst + fcst_width_ratio * (
                                fcst_max - fcst_min))

                    selected_df = day_df[(day_df['Forecast'] >= lower)
                                            & (day_df['Forecast']
                                                <= upper)]
                else:
                    # Nighttime horizons
                    selected_df = pd.DataFrame({'Deviation':np.zeros(1000)})

                try:
                    data = np.ascontiguousarray(selected_df['Deviation'].values)
                    self.marginal_dict[asset, timestep] = ecdf(data)

                except:
                    raise RuntimeError(
                        f'Debugging: unable to fit ECDF for {asset} {timestep}')

    def pca_transform(self, num_of_components):

        self.num_of_components = num_of_components
        self.num_of_hist_data = self.gauss_df.shape[0]

        X = np.concatenate([self.gauss_df[[(asset, i) for i in range(self.num_of_horizons)]].values for asset in self.asset_list])

        # Fit PCA
        pca = PCA(n_components=num_of_components, svd_solver='full')
        Y = pca.fit_transform(X)
        Z = np.concatenate([Y[i*self.num_of_hist_data:(i+1)*self.num_of_hist_data, :] for i,_ in enumerate(self.asset_list)], axis=1)

        pca_gauss_df = pd.DataFrame(data = Z,
                columns=pd.MultiIndex.from_tuples([(asset, comp) for asset in self.asset_list
                    for comp in range(self.num_of_components)]),
                    index=self.gauss_df.index)

        self.pca = pca
        self.pca_residual = 1-pca.explained_variance_ratio_.cumsum()[-1]
        self.pca_gauss_mean, self.pca_gauss_std, self.pca_gauss_df = standardize(pca_gauss_df)

    def fit(self, asset_rho: float, pca_comp_rho: float) -> None:

        if self.num_of_assets == 1:
            pca_comp_prec = graphical_lasso(self.pca_gauss_df, self.num_of_components,
                                           pca_comp_rho)
            asset_prec = np.array([[1.0]])

        elif self.num_of_horizons == 1:
            asset_prec = graphical_lasso(self.pca_gauss_df, self.num_of_assets,
                                         asset_rho)
            pca_comp_prec = np.array([[1.0]])

        else:
            asset_prec, pca_comp_prec = gemini(
                self.pca_gauss_df, self.num_of_assets, self.num_of_components,
                asset_rho, pca_comp_rho
                )

        # compute covariance matrices
        asset_cov = np.linalg.inv(asset_prec)
        self.asset_cov = pd.DataFrame(data=(asset_cov + asset_cov.T) / 2,
                                      index=self.asset_list,
                                      columns=self.asset_list)

        pca_comp_cov = np.linalg.inv(pca_comp_prec)
        pca_comp_indx = ['_'.join(['lag', str(comp)])
                        for comp in range(self.num_of_components)]

        self.horizon_cov = pd.DataFrame(
            data=(pca_comp_cov + pca_comp_cov.T) / 2,
            index=pca_comp_indx, columns=pca_comp_indx
            )

    def generate_gauss_pca_scenarios(self,
            trans_timestep,
            hist_dict,
            nscen: int,
            sqrtcov: Optional[np.array] = None,
            mu: Optional[np.array] = None,
            lower_dict: Optional[pd.Series] = None,
            upper_dict: Optional[pd.Series] = None
            ) -> None:

        if sqrtcov is None:
            sqrtcov = np.kron(sqrtm(self.asset_cov.values).real,
                                sqrtm(self.horizon_cov.values).real)

        # generate random draws from a normal distribution and use the model
        # parameters to transform them into normalized scenario deviations
        arr = sqrtcov @ np.random.randn(
            len(self.asset_list) * self.num_of_components, nscen)

        if mu is not None:
            arr += mu

        pca_scen_gauss_df = pd.DataFrame(
            data=arr.T, columns=pd.MultiIndex.from_tuples(
                [(asset, horizon) for asset in self.asset_list
                 for horizon in range(self.num_of_components)]
                )
            )
        pca_scen_gauss_df = pca_scen_gauss_df * self.pca_gauss_std + self.pca_gauss_mean
        self.pca_scen_gauss_df = pca_scen_gauss_df.copy()

        Y = self.pca.inverse_transform(
            np.concatenate([self.pca_scen_gauss_df[[(asset,c) for c in range(self.num_of_components)]].values
                    for asset in self.asset_list])
                    )
        scen_df = pd.DataFrame(
            data=np.concatenate([Y[i*nscen:(i+1)*nscen, :] for i in range(self.num_of_assets)], axis=1),
            columns=pd.MultiIndex.from_tuples(
                    [(asset, horizon) for asset in self.asset_list
                        for horizon in range(self.num_of_horizons)]
                    )
                )

        self.scen_gauss_df = scen_df.copy()

        scen_df = scen_df * self.gauss_std + self.gauss_mean

        scen_df.columns = pd.MultiIndex.from_tuples(
            scen_df.columns).set_levels(self.scen_timesteps, level=1)

        # Fit conditional marginal distributions
        self.fit_conditional_marginal_dist(trans_timestep, hist_dict)

        # invert the Gaussian scenario deviations by the marginal distributions
        if not self.gauss:

            scen_means, scen_vars = scen_df.mean(), scen_df.std()

            # data considered as point mass if variance < 1e-2
            scen_means[scen_vars<1e-2] = 999999.
            scen_vars[scen_vars<1e-2] = 1.

            u_mat = norm.cdf((scen_df - scen_means) / scen_vars)

            if self.fit_conditional_marginal_dist:
                scen_df = pd.DataFrame({
                    col: self.marginal_dict[col].quantfun(u_mat[:, i])
                    for i, col in enumerate(scen_df.columns)
                    })
            else:
                scen_df = pd.DataFrame({
                    col: qdist(self.gpd_dict[col], u_mat[:, i])
                    for i, col in enumerate(scen_df.columns)
                    })

        # if we have loaded forecasts for the scenario time points, add the
        # unnormalized deviations to the forecasts to produce scenario values
        self.scen_deviation_df = scen_df.copy()
        if self.forecasts is not None:
            scen_df = self.scen_deviation_df + self.forecasts

            if lower_dict is None:
                lower_dict = {site: 0. for site in self.asset_list}
            elif lower_dict == 'devi_min':
                lower_dict = self.load_devi_min

            if upper_dict is None:
                upper_dict = {site: None for site in self.asset_list}

            for site in self.asset_list:
                scen_df[site] = scen_df[site].clip(lower=lower_dict[site],
                                                   upper=upper_dict[site])

            self.scen_df = scen_df

        else:
            self.scen_df = None
