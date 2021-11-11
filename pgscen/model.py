
import numpy as np
import warnings
from scipy.linalg import sqrtm
import pandas as pd
from scipy.stats import norm
from pgscen.utils.r_utils import (qdist, gaussianize, graphical_lasso, gemini,
                                  fit_dist, get_ecdf_data)


def get_asset_list(hist_actual_df, hist_forecast_df):
    assert 'Issue_time' in hist_forecast_df.columns, (
        "hist_forecast_df must have an Issue_time column!")
    assert 'Forecast_time' in hist_forecast_df.columns, (
        "hist_forecast_df must have a Forecast_time column!")

    asset_list = sorted(hist_actual_df.columns)
    assert asset_list == sorted(set(hist_forecast_df.columns)
                                - {'Issue_time', 'Forecast_time'}), (
        "hist_actual_df and hist_forecast_df assets do not match!")

    return asset_list


class GeminiError(Exception):
    pass


class GeminiModel(object):

    def __init__(self, scen_start_time, hist_dfs=None, gauss_df=None,
                 dev_index=None, forecast_resolution_in_minute=60,
                 num_of_horizons=24, forecast_lead_time_in_hour=12):
        self.scen_start_time = scen_start_time
        self.num_of_horizons = num_of_horizons
        self.forecast_resolution_in_minute = forecast_resolution_in_minute
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
            ).tolist()

        if hist_dfs is not None:
            self.gauss = False
            self.asset_list = get_asset_list(hist_dfs['actual'],
                                             hist_dfs['forecast'])

            # Put actual, forecast and deviation into one pandas DataFrame
            hist_df = hist_dfs['actual'].reset_index().merge(
                hist_dfs['forecast'], how='inner', left_on='Time',
                right_on='Forecast_time', suffixes=('_actual','_forecast')
                ).set_index('Time')

            self.deviation_dict = dict()
            for asset in self.asset_list:
                act = hist_df[asset + '_actual']
                fcst = hist_df[asset + '_forecast']

                self.deviation_dict[asset] = pd.DataFrame(
                    {'Actual': act, 'Forecast': fcst, 'Deviation': act - fcst},
                    index=hist_df.index
                    )

            # Compute deviation from historical data: dev = actual - forecast.
            hist_dev_dict = dict()
            for issue_time, fcsts in hist_dfs['forecast'].groupby(
                    'Issue_time'):
                fcst_start_time = issue_time + pd.Timedelta(
                    self.forecast_lead_hours, unit='H')

                fcst_end_time = pd.Timedelta(self.forecast_resolution_in_minute
                                             * (self.num_of_horizons - 1),
                                             unit='min')
                fcst_end_time += fcst_start_time

                # Get actual
                act_df = hist_dfs['actual'][
                    (hist_dfs['actual'].index >= fcst_start_time)
                    & (hist_dfs['actual'].index <= fcst_end_time)
                    ][self.asset_list]

                # Get forecast
                fcst_df = fcsts.drop(columns='Issue_time').set_index(
                    'Forecast_time').sort_index()

                fcst_df = fcst_df[
                    (fcst_df.index >= fcst_start_time)
                    & (fcst_df.index <= fcst_end_time)
                    ][self.asset_list]

                # Create lagged deviations
                if act_df.shape != (self.num_of_horizons,
                                    len(self.asset_list)):
                    warnings.warn(
                        f'unable to find actual data to be matched with '
                        f'forecast issued at {issue_time}',
                        RuntimeWarning
                        )

                elif fcst_df.shape != (self.num_of_horizons,
                                       len(self.asset_list)):
                    warnings.warn(
                        f'forecast issued at {issue_time} does not have '
                        f'{self.num_of_horizons} horizons',
                        RuntimeWarning
                        )

                # Compute difference
                else:
                    hist_dev_dict[
                        fcst_start_time] = act_df.stack() - fcst_df.stack()

                    hist_dev_dict[fcst_start_time].index = hist_dev_dict[
                        fcst_start_time].index.set_levels(
                        tuple(range(self.num_of_horizons)),
                        level=0
                        )

            self.hist_dev_df = pd.DataFrame(hist_dev_dict)
            self.hist_dev_df.index = self.hist_dev_df.index.swaplevel()
            self.hist_dev_df = self.hist_dev_df.sort_index().transpose()

            if dev_index is not None:
                self.hist_dev_df = self.hist_dev_df[
                    self.hist_dev_df.index.isin(dev_index)]

            gpd_dict, self.gauss_df = gaussianize(self.hist_dev_df)
            self.gpd_dict = {
                (asset, timestep): gpd_dict[asset, horizon]
                for asset in self.asset_list
                for horizon, timestep in enumerate(self.scen_timesteps)
                }

        elif gauss_df is not None:
            self.gauss = True
            self.asset_list = gauss_df.columns.levels[0]
            self.gauss_df = gauss_df

        self.num_of_assets = len(self.asset_list)
        self.asset_cov = None
        self.horizon_cov = None
        self.scen_gauss_df = None
        self.scen_deviation_df = None
        self.forecasts = None
        self.conditional_gpd_dict = None
        self.scen_df = None
        #self.load_devi_min = self.hist_dev_df.min()

    def fit(self, asset_rho, horizon_rho):
        if self.num_of_assets == 1:
            # Only one asset run plain GLASSO
            horizon_prec = graphical_lasso(self.gauss_df, self.num_of_horizons,
                                           horizon_rho)
            asset_prec = np.array([[1.0]])

        elif self.num_of_horizons == 1:
            # Only one asset run plain GLASSO
            asset_prec = graphical_lasso(self.gauss_df, self.num_of_assets,
                                         asset_rho)
            horizon_prec = np.array([[1.0]])

        else:
            # Multiple assets and lags, run GEMINI
            asset_prec, horizon_prec = gemini(
                self.gauss_df, self.num_of_assets, self.num_of_horizons,
                asset_rho, horizon_rho
                )

        # Compute covariance matrices
        asset_cov = np.linalg.inv(asset_prec)
        self.asset_cov = pd.DataFrame(data=(asset_cov + asset_cov.T) / 2,
                                      index=self.asset_list,
                                      columns=self.asset_list)

        horizon_cov = np.linalg.inv(horizon_prec)
        horizon_indx = ['_'.join(['lag', str(hz)])
                        for hz in range(self.num_of_horizons)]

        self.horizon_cov = pd.DataFrame(
            data=(horizon_cov + horizon_cov.T) / 2,
            index=horizon_indx, columns=horizon_indx
            )

    def get_forecast(self, forecast_df):
        """
        Get forecast
        """

        use_forecasts = forecast_df[
            forecast_df['Issue_time'] == self.forecast_issue_time].drop(
                columns='Issue_time').set_index('Forecast_time')

        use_forecasts = use_forecasts[
            (use_forecasts.index >= self.scen_start_time)
            & (use_forecasts.index <= self.scen_end_time)
            ].sort_index()
        use_forecasts.index = self.scen_timesteps

        self.forecasts = use_forecasts.unstack()

    def fit_conditional_gpd(self, asset_type, bin_width_ratio=0.05, min_sample_size=200):
        """
        Fit conditional GPD
        """
        assert asset_type == 'load' or asset_type == 'wind'

        self.conditional_gpd_dict = {}
        for asset in self.asset_list:
            asset_df = self.deviation_dict[asset]
            fcst_min = asset_df['Forecast'].min()
            fcst_max = asset_df['Forecast'].max()

            for timestep in self.scen_timesteps:
                fcst = self.forecasts[asset, timestep]

                lower = max(fcst_min,
                            fcst - bin_width_ratio * (fcst_max - fcst_min))
                upper = min(fcst_max,
                            fcst + bin_width_ratio * (fcst_max - fcst_min))

                selected_df = asset_df[(asset_df['Forecast'] >= lower)
                                        & (asset_df['Forecast'] <= upper)]
                data = np.ascontiguousarray(
                    selected_df['Deviation'].values)

                if len(data) < min_sample_size:
                    # If binning data on forecast has < 200 samples, use
                    # the nearest data ponts as samples
                    idx = (asset_df.sort_values(
                        'Forecast') - fcst).abs().sort_values(
                        'Forecast').index[0:min_sample_size]
                    data = np.ascontiguousarray(
                        asset_df.loc[idx, 'Deviation'].values)

                try:
                    self.conditional_gpd_dict[asset, timestep] = fit_dist(data)

                except:
                    raise RuntimeError(
                        f'Debugging: unable to fit gpd for {asset} {timestep}')

    def fit_solar_conditional_gpd(self, trans_horizon, bin_width_ratio=0.05, min_sample_size=200):
        """
        Fit solar conditional GPD 
        """
        
        self.conditional_gpd_dict = {}

        for asset in self.asset_list:
            print(asset)

            sunrise_timestep = trans_horizon['sunrise'][asset]['timestep']
            sunset_timestep = trans_horizon['sunset'][asset]['timestep']
            
            # Data for other horizons
            asset_df = self.deviation_dict[asset]

            asset_df = asset_df[asset_df['Actual'] > 0.]

            fcst_min = asset_df['Forecast'].min()
            fcst_max = asset_df['Forecast'].max()

            for timestep in self.scen_timesteps:   
                  
                if timestep == sunrise_timestep or timestep == sunset_timestep:
                    # Use train data
                    data = np.ascontiguousarray(get_ecdf_data(self.gpd_dict[asset, timestep]))         
                else:
                    # Take fcst +/- 5% bin
                    fcst = self.forecasts[asset, timestep]
                    lower = max(fcst_min, fcst - bin_width_ratio * (
                                    fcst_max - fcst_min))
                    upper = min(fcst_max, fcst + bin_width_ratio * (
                                fcst_max - fcst_min))

                    selected_df = asset_df[(asset_df['Forecast'] >= lower)
                                            & (asset_df['Forecast']
                                                <= upper)]

                    data = np.ascontiguousarray(
                        selected_df['Deviation'].values)

                    if len(data) < min_sample_size:
                        # If binning data on forecast has < 200 samples,
                        # use the nearest data ponts as samples
                        idx = (asset_df.sort_values(
                            'Forecast') - fcst).abs().sort_values(
                            'Forecast').index[0:min_sample_size]
                        data = np.ascontiguousarray(
                            asset_df.loc[idx, 'Deviation'].values)
                        
                try:
                    self.conditional_gpd_dict[asset, timestep] = fit_dist(data)

                except:
                    raise RuntimeError(
                        f'Debugging: unable to fit gpd for {asset} {timestep}')


    def generate_gauss_scenarios(self, nscen: int, sqrt_cov=None, mu=None,
                                 lower_dict=None, upper_dict=None):
        """
        Generate conditional or unconditional Gaussian scenarios.

        :param nscen: number of scenarios
        :type nscen: int
        :param conditional: whether to generate conditional Gaussian scenarios
                            if true, use sqrtcov and mu in kwargs arguments
                            otherwise use self.asset_cov and self.lag_cov.
                            defaults to false
        :type conditional: boolean
        :param kwargs:
            sqrtcov: 2d numpy array, square root of the covariance matrix
            mu: 1d numpy array, mean vector

        """
        if self.gauss and (lower_dict is not None or upper_dict is not None):
            print("Scenario deviations cannot be clipped "
                  "when given in gaussian form!")

        if self.asset_cov is None or self.horizon_cov is None:
            raise GeminiError("Cannot generate scenarios with a model"
                              "that has not been fit yet!")

        if sqrt_cov is None:
            sqrt_cov = np.kron(sqrtm(self.asset_cov.values).real,
                               sqrtm(self.horizon_cov.values).real)

        arr = sqrt_cov @ np.random.randn(
            len(self.asset_list) * self.num_of_horizons, nscen)
        if mu is not None:
            arr += mu

        scen_df = pd.DataFrame(
            data=arr.T, columns=pd.MultiIndex.from_tuples(
                [(asset, horizon) for asset in self.asset_list
                 for horizon in range(self.num_of_horizons)]
                )
            )

        self.scen_gauss_df = scen_df.copy()
        scen_df.columns = pd.MultiIndex.from_tuples(
            scen_df.columns).set_levels(self.scen_timesteps, level=1)

        # Invert the Gaussian scenarios by the marginal distributions.
        if not self.gauss:
            scen_means, scen_vars = scen_df.mean(), scen_df.std()
            u_mat = norm.cdf((scen_df - scen_means) / scen_vars)

            if self.conditional_gpd_dict:
                scen_df = pd.DataFrame({
                    col: qdist(self.conditional_gpd_dict[col], u_mat[:, i])
                    for i, col in enumerate(scen_df.columns)
                    })
            else:
                scen_df = pd.DataFrame({
                    col: qdist(self.gpd_dict[col], u_mat[:, i])
                    for i, col in enumerate(scen_df.columns)
                    })

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

    def conditional_multivar_normal_partial_time(
            self, cond_hz_start, cond_hz_end, cond_scen_df):
        """
        Compute mean and the squre root of the covariance matrix
        of a multivariate Gaussian distribution conditioned on a set of
        realizations of the data for a certain time interval and
        for all assets.

        For example, suppose a gemini model is fitted for a set of assets
        for the lags from 0 to 23. If the scenarios for all assets and
        lags from 8 to 17 have been generated, this function computes the
        covariance matrix and mean of the Gaussian distribution condtioned on
        scenarios for lag 8 to 17 havbe been realized.

        :param condition_scen_lag_start: first lag for which scenarios have been realized
        :type condition_scen_lag_start: int
        :param condition_scen_lag_end: last lag for which scenarios have been realized
        :type condition_scen_lag_end: int
        :param condition_scen_df: scenarios for all assets and
                                  for all lags from ``condition_scen_lag_start`` to ``condition_scen_lag_end``
        :type condition_scen_df: pandas DataFrame

        :return: sqrtcov -- square root of the covariance matrix
                 mu -- mean vector

        """
        use_scen_df = cond_scen_df[
            [(asset, horizon) for asset in self.asset_list
             for horizon in range(cond_hz_start, cond_hz_end + 1)]
            ]

        r_mat = np.zeros((self.num_of_horizons, self.num_of_horizons))
        r_mat[cond_hz_start:(cond_hz_end + 1),
              cond_hz_start:(cond_hz_end + 1)] = np.linalg.inv(
                self.horizon_cov.values[cond_hz_start:(cond_hz_end + 1),
                                        cond_hz_start:(cond_hz_end + 1)]
                )

        u_mat = np.eye(self.num_of_horizons) - self.horizon_cov.values @ r_mat
        sqrtcov = np.kron(sqrtm(self.asset_cov.values).real,
                          sqrtm(u_mat @ self.horizon_cov.values
                                @ u_mat.T).real)

        r_diag = np.kron(
            np.eye(self.num_of_assets),
            self.horizon_cov.values @ r_mat[:, cond_hz_start:(cond_hz_end + 1)]
            )
        mu = r_diag @ use_scen_df.values.T

        return sqrtcov.real, mu

    def conditional_multivar_normal_aggregation(self,
                                                aggregates_df, membership):
        """
        Compute mean and a covariance matrix of a multivariate
        Gaussian distribution conditioned on a set of realizations
        of the aggregations (sum) of random variables.

        For example, suppose a gemini model is fitted for a set of assets
        for the lags from 0 to 23. If the scenarios for all assets and
        lags from 8 to 17 have been generated, this function computes the
        covariance matrix and mean of the Gaussian distribution condtioned on
        scenarios for lag 8 to 17 havbe been realized.

        For example, suppose a gemini model if fitted for 5 assets
        and for lags from 0 to 23. Assuming that for all lags from 0 to 23,
        the scenarios of the sum of the first 3 assets and the sum of
        the last 2 assets have been generated. Conditioned on these scenarios,
        this function computes the covariance and mean of the new Gaussian distribution.

        :param num_of_aggregates: number of aggregates
        :type num_of_aggregates: int
        :param aggregates_list: list of aggregates' names
        :type aggregates_list: list of str
        :param membership: dictionary of {aggregate: list of assets)
        :type membership: dict
        :param condition_scen_df: scenarios for all aggregates and for all lags
        :type condition_scen_df: pandas DataFrame

        :return: sqrtcov -- square root of the covariance matrix
                 mu -- mean vector

        """
        aggregates_list = aggregates_df.columns.unique(0).tolist()
        s_mat = np.zeros((len(aggregates_list), len(self.asset_list)))

        for aggregate, assets in membership.items():
            agg_indx = aggregates_list.index(aggregate)

            for asset in assets:
                s_mat[agg_indx, self.asset_list.index(asset)] = 1.

        r_mat = self.asset_cov.values @ s_mat.T @ np.linalg.inv(
            s_mat @ self.asset_cov.values @ s_mat.T)
        mu = np.kron(r_mat,
                     np.eye(self.num_of_horizons)) @ aggregates_df.values.T

        r_mat = np.eye(len(self.asset_list)) - r_mat @ s_mat
        sqrtcov = np.kron(sqrtm(r_mat @ self.asset_cov.values @ r_mat.T).real,
                          sqrtm(self.horizon_cov.values).real)

        return sqrtcov.real, mu
