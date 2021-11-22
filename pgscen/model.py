"""Implementations of graphical LASSO models for use in scenario generation."""

import numpy as np
import pandas as pd

import warnings
from scipy.linalg import sqrtm
from scipy.stats import norm
import rpy2
from typing import List, Dict, Tuple, Iterable, Optional

from pgscen.utils.r_utils import (qdist, gaussianize, graphical_lasso, gemini,
                                  fit_dist, get_ecdf_data)


def get_asset_list(hist_actual_df : pd.DataFrame,
                   hist_forecast_df: pd.DataFrame) -> List[str]:
    """
    Utility for checking actual and forecast dataset format and integrity.

    This function ensures that actual and forecast value dataframes have the
    correct column names, and that these column names match one another.

    Returns
    -------
        asset_list
            The assets that these datasets contain information for.

    """
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
    """
    A class for fitting GEMINI models to asset actual and forecast values and
    using them to generate scenarios for a given time frame.

    Attributes
    ----------
        scen_start_time : pd.TimeStamp
            When the generated scenarios will start.
        num_of_horizons : int
            How many forecast time intervals to generate scenarios for.
        forecast_resolution_in_minute : int
            The frequency of the intervals at which forecasts are provided.
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

        gauss : bool
            Whether this model instance was instantiated used normalized
            historical deviations instead of actual and forecasted values.
        asset_list : List[str]
            The assets for which scenarios will be generated.
        num_of_assets : int
            How many assets there are.

        deviation_dict : Dict[str, pd.DataFrame]
            The historical actuals and forecasts used to instantiate the model
            and the corresponding deviations for each asset.
        hist_dev_df : pd.DataFrame
            The historical deviations indexed by forecast issue time and time
            horizon for each asset.
        gauss_df : pd.DataFrame
            The deviations in `hist_dev_df` normalized using a Gaussian.
        gpd_dict : Dict[pd.Timestamp, rpy2.SignatureTranslatedFunction]
            The parameters of the Gaussian distributions used to turn
            `hist_dev_df` into `gauss_df`

        asset_cov, horizon_cov : Optional[pd.DataFrame]
            Fitted GEMINI model parameters describing covariances between
            asset and time points respectively.

        scen_gauss_df : Optional[pd.DataFrame]
            Generated scenario deviations normed to a Gaussian distribution.
        scen_deviation_df : Optional[pd.DataFrame]
            Unnormalized deviations for generated scenarios.
        forecasts : Optional[Dict[pd.Series]]
            The forecasted values for the scenario time window which were used
            as a basis to generate scenarios.
        conditional_gpd_dict : Optional[Dict[pd.Timestamp,
                                        rpy2.SignatureTranslatedFunction]
            Parameters of the Gaussian distributions used to fit a scenario
            generation model conditional on similar forecast values.

        scen_df : Optional[pd.DataFrame]
            Scenarios generated by this model, with rows corresponding to
            scenarios and columns corresponding to asset x time point.

    """

    def __init__(self,
                 scen_start_time: pd.Timestamp,
                 hist_dfs: Optional[Dict[str, pd.DataFrame]] = None,
                 gauss_df: Optional[pd.DataFrame] = None,
                 dev_index: Optional[Iterable[pd.Timestamp]] = None,
                 forecast_resolution_in_minute: int = 60,
                 num_of_horizons: int = 24,
                 forecast_lead_time_in_hour: int = 12) -> None:
        """
        GeminiModel classes can be instantiated by either giving a pair of
        dataframes with actual and forecasted historical values in a dictionary
        formatted as {'actual': ... , 'forecast': ...} (`hist_dfs`), or by
        passing a pre-computed set of normalized historical deviations in a
        dataframe (`gauss_df`).

        Also note that `dev_index` can be used to choose a subset of historical
        forecast issue times instead of using all available issue times.

        """
        self.scen_start_time = scen_start_time
        self.num_of_horizons = num_of_horizons
        self.forecast_resolution_in_minute = forecast_resolution_in_minute
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

        # if historical data is given, make sure it is formatted correctly
        if hist_dfs is not None:
            self.gauss = False
            self.asset_list = get_asset_list(hist_dfs['actual'],
                                             hist_dfs['forecast'])

            # put actual, forecast and deviation into one dataframe, start by
            # putting actuals and forecasts in the same df
            hist_df = hist_dfs['actual'].reset_index().merge(
                hist_dfs['forecast'], how='inner', left_on='Time',
                right_on='Forecast_time', suffixes=('_actual','_forecast')
                ).set_index('Time')

            # compute deviation from historical data: dev = actual - forecast
            self.deviation_dict = dict()
            for asset in self.asset_list:
                act = hist_df[asset + '_actual']
                fcst = hist_df[asset + '_forecast']

                self.deviation_dict[asset] = pd.DataFrame(
                    {'Actual': act, 'Forecast': fcst, 'Deviation': act - fcst},
                    index=hist_df.index
                    )

            # for each forecast issue time, find the actual values for the time
            # points for which the forecast were made
            hist_dev_dict = dict()
            for issue_time, fcsts in hist_dfs['forecast'].groupby(
                    'Issue_time'):
                fcst_start_time = issue_time + pd.Timedelta(
                    self.forecast_lead_hours, unit='H')

                fcst_end_time = pd.Timedelta(self.forecast_resolution_in_minute
                                             * (self.num_of_horizons - 1),
                                             unit='min')
                fcst_end_time += fcst_start_time

                act_df = hist_dfs['actual'][
                    (hist_dfs['actual'].index >= fcst_start_time)
                    & (hist_dfs['actual'].index <= fcst_end_time)
                    ][self.asset_list]

                fcst_df = fcsts.drop(columns='Issue_time').set_index(
                    'Forecast_time').sort_index()

                fcst_df = fcst_df[
                    (fcst_df.index >= fcst_start_time)
                    & (fcst_df.index <= fcst_end_time)
                    ][self.asset_list]

                # create lagged deviations
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

                # compute difference
                else:
                    hist_dev_dict[
                        fcst_start_time] = act_df.stack() - fcst_df.stack()

                    hist_dev_dict[fcst_start_time].index = hist_dev_dict[
                        fcst_start_time].index.set_levels(
                        tuple(range(self.num_of_horizons)),
                        level=0
                        )

            # create a data frame where the columns are asset name x forecast
            # time horizon, and index is forecast issue times
            self.hist_dev_df = pd.DataFrame(hist_dev_dict)
            self.hist_dev_df.index = self.hist_dev_df.index.swaplevel()
            self.hist_dev_df = self.hist_dev_df.sort_index().transpose()

            # only use deviations from particular dates if given
            if dev_index is not None:
                self.hist_dev_df = self.hist_dev_df[
                    self.hist_dev_df.index.isin(dev_index)]

            # normalize historical deviations using Gaussian distribution
            gpd_dict, self.gauss_df = gaussianize(self.hist_dev_df)
            self.gpd_dict = {
                (asset, timestep): gpd_dict[asset, horizon]
                for asset in self.asset_list
                for horizon, timestep in enumerate(self.scen_timesteps)
                }

        # if normalized historical deviations are given, just use those
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

    def fit(self, asset_rho: float, horizon_rho: float) -> None:
        """
        This function creates and fits a scenario model using historical asset
        values. The model will estimate the distributions of the deviations
        from actual values observed in the forecast dataset.

        Note that a glasso model will be fit if there is only one asset or one
        time point; otherwise, a GEMINI model will be used.

        Arguments
        ---------
            asset_rho
                Regularization hyper-parameter governing asset precisions.
            horizon_rho
                Regularization hyper-parameter governing time point precisions.

        """
        if self.num_of_assets == 1:
            horizon_prec = graphical_lasso(self.gauss_df, self.num_of_horizons,
                                           horizon_rho)
            asset_prec = np.array([[1.0]])

        elif self.num_of_horizons == 1:
            asset_prec = graphical_lasso(self.gauss_df, self.num_of_assets,
                                         asset_rho)
            horizon_prec = np.array([[1.0]])

        else:
            asset_prec, horizon_prec = gemini(
                self.gauss_df, self.num_of_assets, self.num_of_horizons,
                asset_rho, horizon_rho
                )

        # compute covariance matrices
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

    def get_forecast(self, forecast_df: pd.DataFrame) -> None:
        """Find subset of given forecasts for this model's time points."""

        use_forecasts = forecast_df[
            forecast_df['Issue_time'] == self.forecast_issue_time].drop(
                columns='Issue_time').set_index('Forecast_time')

        use_forecasts = use_forecasts[
            (use_forecasts.index >= self.scen_start_time)
            & (use_forecasts.index <= self.scen_end_time)
            ].sort_index()
        use_forecasts.index = self.scen_timesteps

        self.forecasts = use_forecasts.unstack()

    def fit_conditional_gpd(self,
                            asset_type: str, bin_width_ratio: float = 0.05,
                            min_sample_size: int = 200,
                            positive_actual: bool = False) -> None:
        """
        This function fits a conditional GPD using the historical deviations
        for time points whose forecasted values are similar to the forecasted
        values for the time point at which scenarios are to be generated.

        Arguments
        ---------
        asset_type
            The type of asset for which models are being fitted. Must be one of
            "load", "solar", or "wind".
        bin_width_ratio
            The range of historical values which will be considered "similar"
            to the current forecast value, expressed as a proportion of the
            range of these values.
        min_sample_size
            If the number of historical values found using the binning
            criterion described above does not meet this threshold, we will
            append the closest historical values outside of the bin to
            satisfy it.
        positive_actual
            Whether to only consider time points with strictly positive actual
            values, useful especially for solar models with many zero values
            which can cause problems for the model.

        """
        self.conditional_gpd_dict = {}
        for asset in self.asset_list:
            asset_df = self.deviation_dict[asset]

            if positive_actual:
                asset_df = asset_df[asset_df['Actual'] > 0.]

            # find the range of historical forecasts
            fcst_min = asset_df['Forecast'].min()
            fcst_max = asset_df['Forecast'].max()

            for timestep in self.scen_timesteps:
                fcst = self.forecasts[asset, timestep]

                if asset_type == 'load' or asset_type == 'wind':
                    lower = max(fcst_min,
                                fcst - bin_width_ratio * (fcst_max - fcst_min))
                    upper = min(fcst_max,
                                fcst + bin_width_ratio * (fcst_max - fcst_min))

                    selected_df = asset_df[(asset_df['Forecast'] >= lower)
                                           & (asset_df['Forecast'] <= upper)]
                    data = np.ascontiguousarray(
                        selected_df['Deviation'].values)

                    if len(data) < min_sample_size:
                        idx = (asset_df.sort_values(
                            'Forecast') - fcst).abs().sort_values(
                            'Forecast').index[0:min_sample_size]
                        data = np.ascontiguousarray(
                            asset_df.loc[idx, 'Deviation'].values)

                elif asset_type == 'solar':
                    hist_dev_data = get_ecdf_data(
                        self.gpd_dict[asset, timestep])
                    hist_dev_range = np.max(
                        hist_dev_data) - np.min(hist_dev_data)

                    # if the forecasted value is too small, use all historical
                    # deviations
                    if (fcst <= 0.15 * fcst_max
                            and hist_dev_range <= 0.15 * fcst_max):
                        data = np.ascontiguousarray(hist_dev_data)

                    # otherwise take the usual forecast +/- 5% of range bin
                    else:
                        lower = max(fcst_min, fcst - bin_width_ratio * (
                                    fcst_max - fcst_min))
                        upper = min(fcst_max, fcst + bin_width_ratio * (
                                    fcst_max - fcst_min))

                        selected_df = asset_df[(asset_df['Forecast'] >= lower)
                                               & (asset_df['Forecast']
                                                  <= upper)]

                        data = np.ascontiguousarray(
                            selected_df['Deviation'].values)

                        # if binning data on forecast has insufficient samples,
                        # use the nearest data points as samples
                        if len(data) < min_sample_size:
                            idx = (asset_df.sort_values(
                                'Forecast') - fcst).abs().sort_values(
                                'Forecast').index[0:min_sample_size]
                            data = np.ascontiguousarray(
                                asset_df.loc[idx, 'Deviation'].values)

                else:
                    raise RuntimeError(f'Unrecognizable asset_type '
                                       f'{asset_type}, must be one of load, '
                                       f'solar or wind')

                try:
                    self.conditional_gpd_dict[asset, timestep] = fit_dist(data)

                except:
                    raise RuntimeError(
                        f'Debugging: unable to fit gpd for {asset} {timestep}')

    def generate_gauss_scenarios(
            self,
            nscen: int, sqrt_cov: Optional[np.array] = None,
            mu: Optional[np.array] = None,
            lower_dict: Optional[pd.Series] = None,
            upper_dict: Optional[pd.Series] = None
            ) -> None:
        """
        Generate conditional or unconditional Gaussian scenarios.

        Arguments
        ---------
            nscen
                The number of scenarios to generate.
            sqrt_cov
                Pre-computed covariances if generating conditional scenarios.
                Otherwise the model will use `self.asset_cov` and
                `self.horizon_cov`.
            mu
                Pre-computed means if generating conditional scenarios.
            lower_dict, upper_dict
                If given, clips generated scenarios to this lower/upper bound.

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

        # generate random draws from a normal distribution and use the model
        # parameters to transform them into normalized scenario deviations
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

        # invert the Gaussian scenario deviations by the marginal distributions
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

    def conditional_multivar_normal_partial_time(
            self,
            cond_hz_start: int, cond_hz_end: int, cond_scen_df: pd.DataFrame
            ) -> Tuple[np.array, np.array]:
        """
        Compute mean and the square root of the covariance matrix
        of a multivariate Gaussian distribution conditioned on a set of
        realizations of the data for a certain time interval and
        for all assets.

        For example, suppose a gemini model is fitted for a set of assets
        for the lags from 0 to 23. If the scenarios for all assets and
        lags from 8 to 17 have been generated, this function computes the
        covariance matrix and mean of the Gaussian distribution condtioned on
        scenarios for lag 8 to 17 have been realized.

        Arguments
        ---------
            cond_hz_start, cond_hz_end
                First and last time points for which scenarios
                have been realized.

            cond_scen_df
                Scenarios for all assets and for all time points from
                `cond_hz_start`` to `cond_hz_end`.

        Returns
        -------
            sqrtcov
                The square root of the covariance matrix.
            mu
                Vector of means.

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

    def conditional_multivar_normal_aggregation(
            self,
            aggregates_df: pd.DataFrame, membership: Dict[str, List[str]]
            )-> Tuple[np.array, np.array]:
        """
        Compute mean and a covariance matrix of a multivariate Gaussian
        distribution conditioned on a set of realizations of the aggregations
        (sum) of random variables.

        For example, suppose a gemini model is fitted for a set of assets for
        the lags from 0 to 23. If the scenarios for all assets and lags from 8
        to 17 have been generated, this function computes the covariance matrix
        and mean of the Gaussian distribution conditioned on scenarios for lag
        8 to 17 havbe been realized.

        For example, suppose a gemini model if fitted for 5 assets and for lags
        from 0 to 23. Assuming that for all lags from 0 to 23, the scenarios of
        the sum of the first 3 assets and the sum of the last 2 assets have
        been generated. Conditioned on these scenarios, this function computes
        the covariance and mean of the new Gaussian distribution.

        Arguments
        ---------
            aggregates_df
                Scenarios for all aggregate assets and all time points.
            membership
                Which individual assets correspond to which aggregated assets.

        Returns
        -------
            sqrtcov
                The square root of the covariance matrix.
            mu
                Vector of means.

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
