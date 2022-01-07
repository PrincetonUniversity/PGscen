from pgscen.model import GeminiModel
from pgscen.engine import GeminiEngine

import numpy as np
import pandas as pd

import warnings
from scipy.linalg import sqrtm
from scipy.stats import norm
from typing import List, Dict, Tuple, Iterable, Optional

from pgscen.utils.r_utils import (qdist, gaussianize, graphical_lasso, gemini,
                                  fit_dist, get_ecdf_data)

from sklearn.decomposition import PCA                                  

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

    def fit(self, num_of_components: float, asset_rho: float, horizon_rho: float) -> None:

        self.model = PCAGeminiModel(self.scen_start_time, self.get_hist_df_dict(),
                                 None, None,
                                 self.forecast_resolution_in_minute,
                                 self.num_of_horizons,
                                 self.forecast_lead_hours)
        self.model.pca_transform(num_of_components)
        self.model.fit(asset_rho, horizon_rho)
        

    def create_scenario(self,
                        nscen: int, forecast_df: pd.DataFrame,
                        **gpd_args) -> None:

        self.model.get_forecast(forecast_df)
        self.model.generate_gauss_pca_scenarios(nscen, upper_dict=self.meta_df.AC_capacity_MW)

        self.scenarios[self.asset_type] = self.model.scen_df
        self.forecasts[self.asset_type] = self.get_forecast(forecast_df)

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


    # def pca_transform(self, num_of_components):
    #     self.num_of_components = num_of_components
    #     self.num_of_hist_data = self.hist_dev_df.shape[0]

    #     mean_dict = dict()
    #     X_centered = np.zeros((self.num_of_assets*self.num_of_hist_data,24))

    #     for i,asset in enumerate(self.asset_list):
    #         cols = [(asset,h) for h in range(24)]
            
    #         X = self.hist_dev_df[cols].values
            
    #         mean_dict[asset] = dict()
    #         mean_dict[asset]['original'] = X
    #         mean_dict[asset]['mean'] = X.mean(axis=0)
            
    #         X_centered[i*self.num_of_hist_data:(i+1)*self.num_of_hist_data,:] = X-X.mean(axis=0)
            
    #     pca = PCA(n_components=num_of_components, svd_solver='full')
    #     Y = pca.fit_transform(X_centered)

    #     arr = np.zeros((self.num_of_hist_data,self.num_of_assets*self.num_of_components))

    #     for i,asset in enumerate(self.asset_list):
    #         arr[:,i*self.num_of_components:(i+1)*self.num_of_components] = Y[i*self.num_of_hist_data:(i+1)*self.num_of_hist_data,:]
            
    #     self.pca_hist_dev_df = pd.DataFrame(data=arr, 
    #             columns=pd.MultiIndex.from_tuples([(asset, comp) for asset in self.asset_list
    #                 for comp in range(self.num_of_components)]), 
    #                 index=self.hist_dev_df.index)
    #     self.hist_dev_mean_dict = mean_dict
    #     self.pca = pca
    #     self.pca_residual = 1-pca.explained_variance_ratio_.cumsum()[-1]

    #     # Make transformed data Gaussian
    #     self.pca_scen_timesteps = self.scen_timesteps[0:self.num_of_components]

    #     gpd_dict, self.gauss_df = gaussianize(self.pca_hist_dev_df)
    #     self.gpd_dict = {
    #         (asset, timestep): gpd_dict[asset, horizon]
    #         for asset in self.asset_list
    #         for horizon, timestep in enumerate(self.pca_scen_timesteps)
    #         }

    def pca_transform(self, num_of_components):
        """
        Fit one PCA per asset
        """

        self.num_of_components = num_of_components
        self.num_of_hist_data = self.hist_dev_df.shape[0]

        pca_dict = dict()
        pca_data = np.zeros((self.num_of_hist_data,self.num_of_assets*self.num_of_components))

        for i,asset in enumerate(self.asset_list):
            cols = [(asset,h) for h in range(24)]
            
            X = self.hist_dev_df[cols].values
            
            pca_dict[asset] = dict()
            pca_dict[asset]['data'] = X
            
            pca = PCA(n_components=num_of_components, svd_solver='full')
            pca_data[:, i*self.num_of_components:(i+1)*self.num_of_components] = pca.fit_transform(X)
            pca_dict[asset]['pca'] = pca
            

        self.pca_dict = pca_dict
        self.pca_hist_dev_df = pd.DataFrame(data=pca_data, 
                columns=pd.MultiIndex.from_tuples([(asset, comp) for asset in self.asset_list
                    for comp in range(self.num_of_components)]), 
                    index=self.hist_dev_df.index)
                    
        # Make transformed data Gaussian
        self.pca_scen_timesteps = self.scen_timesteps[0:self.num_of_components]

        gpd_dict, self.gauss_df = gaussianize(self.pca_hist_dev_df)
        self.gpd_dict = {
            (asset, timestep): gpd_dict[asset, horizon]
            for asset in self.asset_list
            for horizon, timestep in enumerate(self.pca_scen_timesteps)
            }


    def fit(self, asset_rho: float, pca_comp_rho: float) -> None:
        if self.num_of_assets == 1:
            horizon_prec = graphical_lasso(self.gauss_df, self.num_of_components,
                                           pca_comp_rho)
            asset_prec = np.array([[1.0]])

        elif self.num_of_horizons == 1:
            asset_prec = graphical_lasso(self.gauss_df, self.num_of_assets,
                                         asset_rho)
            horizon_prec = np.array([[1.0]])

        else:
            asset_prec, pca_comp_prec = gemini(
                self.gauss_df, self.num_of_assets, self.num_of_components,
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


    # def generate_gauss_pca_scenarios(self, 
    #         nscen: int,
    #         lower_dict: Optional[pd.Series] = None,
    #         upper_dict: Optional[pd.Series] = None
    #         ) -> None:
            
    #     sqrt_cov = np.kron(sqrtm(self.asset_cov.values).real,
    #                         sqrtm(self.horizon_cov.values).real)

    #     # generate random draws from a normal distribution and use the model
    #     # parameters to transform them into normalized scenario deviations
    #     arr = sqrt_cov @ np.random.randn(
    #         len(self.asset_list) * self.num_of_components, nscen)

    #     scen_df = pd.DataFrame(
    #         data=arr.T, columns=pd.MultiIndex.from_tuples(
    #             [(asset, horizon) for asset in self.asset_list
    #              for horizon in range(self.num_of_components)]
    #             )
    #         )

    #     self.scen_gauss_df = scen_df.copy()
    #     scen_df.columns = pd.MultiIndex.from_tuples(
    #         scen_df.columns).set_levels(self.pca_scen_timesteps, level=1)

    #     # invert the Gaussian scenario deviations by the marginal distributions
    #     if not self.gauss:
    #         scen_means, scen_vars = scen_df.mean(), scen_df.std()
    #         u_mat = norm.cdf((scen_df - scen_means) / scen_vars)

    #         if self.conditional_gpd_dict:
    #             scen_df = pd.DataFrame({
    #                 col: qdist(self.conditional_gpd_dict[col], u_mat[:, i])
    #                 for i, col in enumerate(scen_df.columns)
    #                 })
    #         else:
    #             scen_df = pd.DataFrame({
    #                 col: qdist(self.gpd_dict[col], u_mat[:, i])
    #                 for i, col in enumerate(scen_df.columns)
    #                 })

    #     self.scen_pca_df = scen_df

    #     # inverse pca transform
    #     arr = np.zeros((nscen, self.num_of_horizons * self.num_of_assets))
    #     for i,asset in enumerate(self.asset_list):
    #         cols = [(asset,t) for t in self.pca_scen_timesteps]
    #         arr[:, (i * self.num_of_horizons):((i+1) * self.num_of_horizons)] = \
    #             self.pca.inverse_transform(self.scen_pca_df[cols].values) + \
    #                 self.hist_dev_mean_dict[asset]['mean']
    #     scen_df = pd.DataFrame(data=arr, columns=pd.MultiIndex.from_tuples(
    #                 [(asset, ts) for asset in self.asset_list
    #                     for ts in self.scen_timesteps]
    #                 )
    #             )

    #     # if we have loaded forecasts for the scenario time points, add the
    #     # unnormalized deviations to the forecasts to produce scenario values
    #     self.scen_deviation_df = scen_df.copy()
    #     if self.forecasts is not None:
    #         scen_df = self.scen_deviation_df + self.forecasts

    #         if lower_dict is None:
    #             lower_dict = {site: 0. for site in self.asset_list}
    #         elif lower_dict == 'devi_min':
    #             lower_dict = self.load_devi_min

    #         if upper_dict is None:
    #             upper_dict = {site: None for site in self.asset_list}

    #         for site in self.asset_list:
    #             scen_df[site] = scen_df[site].clip(lower=lower_dict[site],
    #                                                upper=upper_dict[site])

    #         self.scen_df = scen_df

    #     else:
    #         self.scen_df = None


    """
    Generate scenarios for the case that each asset has its own pca.
    """
    def generate_gauss_pca_scenarios(self, 
            nscen: int,
            lower_dict: Optional[pd.Series] = None,
            upper_dict: Optional[pd.Series] = None
            ) -> None:

        sqrt_cov = np.kron(sqrtm(self.asset_cov.values).real,
                            sqrtm(self.horizon_cov.values).real)

        # generate random draws from a normal distribution and use the model
        # parameters to transform them into normalized scenario deviations
        arr = sqrt_cov @ np.random.randn(
            len(self.asset_list) * self.num_of_components, nscen)

        scen_df = pd.DataFrame(
            data=arr.T, columns=pd.MultiIndex.from_tuples(
                [(asset, horizon) for asset in self.asset_list
                 for horizon in range(self.num_of_components)]
                )
            )

        self.scen_gauss_df = scen_df.copy()
        scen_df.columns = pd.MultiIndex.from_tuples(
            scen_df.columns).set_levels(self.pca_scen_timesteps, level=1)

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

        self.scen_pca_df = scen_df

        # inverse pca transform
        arr = np.zeros((nscen, self.num_of_horizons * self.num_of_assets))
        for i,asset in enumerate(self.asset_list):
            cols = [(asset,t) for t in self.pca_scen_timesteps]
            arr[:, (i * self.num_of_horizons):((i+1) * self.num_of_horizons)] = \
                self.pca_dict[asset]['pca'].inverse_transform(self.scen_pca_df[cols].values)
        scen_df = pd.DataFrame(data=arr, columns=pd.MultiIndex.from_tuples(
                    [(asset, ts) for asset in self.asset_list
                        for ts in self.scen_timesteps]
                    )
                )

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


            
