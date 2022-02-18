from os import times
from pgscen.model import GeminiModel
from pgscen.engine import GeminiEngine

import numpy as np
import pandas as pd

import warnings
from scipy.linalg import sqrtm
from scipy.stats import norm
from typing import List, Dict, Tuple, Iterable, Optional

from pgscen.utils.r_utils import (qdist, gaussianize, graphical_lasso, gemini, 
                                  fit_dist, get_ecdf_data, ecdf)
from pgscen.utils.solar_utils import (get_yearly_date_range, get_asset_trans_hour_info)

from sklearn.decomposition import PCA        
from astral import LocationInfo                        

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

        ################### Compute transitional hour delay time #########################

        print('computing hour delay time....')

        hist_dates = solar_hist_forecast_df.groupby('Issue_time').head(1)['Forecast_time'].tolist()
        delay_dict = dict()
        hist_sun_dict = dict()

        for asset,row in self.meta_df.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            loc = LocationInfo(asset,'Texas','USA',lat,lon)

            act_df = solar_hist_actual_df[asset]
            fcst_df = solar_hist_forecast_df.set_index('Forecast_time')[asset]
            
            sunrise_data,sunset_data = [],[]
            day_data = {'Time':[], 'Actual':[], 'Forecast':[], 'Deviation':[]}

            for date in hist_dates:
                trans = get_asset_trans_hour_info(loc,date)
                
                # Sunrise
                sunrise_dict = trans['sunrise']
                sunrise_time,sunrise_active,sunrise_horizon = sunrise_dict['time'],\
                    sunrise_dict['active'],sunrise_dict['timestep']
                act = act_df.loc[sunrise_horizon]
                fcst = fcst_df.loc[sunrise_horizon]
                sunrise_data.append([sunrise_time,sunrise_horizon,act,fcst,act-fcst,sunrise_active])
                
                # Sunset
                sunset_dict = trans['sunset']
                sunset_time,sunset_active,sunset_horizon = sunset_dict['time'],\
                    sunset_dict['active'],sunset_dict['timestep']
                act = act_df.loc[sunset_horizon]
                fcst = fcst_df.loc[sunset_horizon]
                sunset_data.append([sunset_time,sunset_horizon,act,fcst,act-fcst,sunset_active])

                # Daytime
                act = act_df[(act_df.index>sunrise_horizon) & (act_df.index<sunset_horizon)].sort_index()
                fcst = fcst_df[(fcst_df.index>sunrise_horizon) & (fcst_df.index<sunset_horizon)].sort_index()
                day_data['Time'] += act.index.tolist()
                day_data['Actual'] += act.values.tolist()
                day_data['Forecast'] += fcst.values.tolist()
                day_data['Deviation'] += (act-fcst).values.tolist()



            sunrise_df = pd.DataFrame(data=sunrise_data,
                                    columns=['Time','Horizon','Actual','Forecast','Deviation','Active Minutes'])
            sunset_df = pd.DataFrame(data=sunset_data,
                                    columns=['Time','Horizon','Actual','Forecast','Deviation','Active Minutes'])
            day_df = pd.DataFrame(day_data).set_index('Time')
            
            hist_sun_dict[asset] = {'sunrise':sunrise_df, 'sunset':sunset_df, 'day':day_df,}

            # Figure out delay times
            for m in range(60):
                if sunrise_df[sunrise_df['Active Minutes']==m]['Actual'].sum()>0:
                    sunrise_delay_in_minutes = m-1
                    break
                    
            for m in range(60):
                if sunset_df[sunset_df['Active Minutes']==m]['Actual'].sum()>0:
                    sunset_delay_in_minutes = m-1
                    break 
                    
            delay_dict[asset] = {'sunrise':sunrise_delay_in_minutes,'sunset':sunset_delay_in_minutes}
        
        self.hist_sun_info = hist_sun_dict
        self.trans_delay = delay_dict


        ################################### Compute transitional hour statistics ######################

        trans_horizon_dict = {'sunrise':{},'sunset':{}}
        
        for asset,row in self.meta_df.iterrows():
            
            lat = row['latitude']
            lon = row['longitude']
            loc = LocationInfo(asset,'Texas','USA',lat,lon)
            
            sunrise_delay_in_minutes = self.trans_delay[asset]['sunrise']
            sunset_delay_in_minutes = self.trans_delay[asset]['sunset']   

            ################# Get scenario date transitional hour timestep ###############
            trans = get_asset_trans_hour_info(loc,self.scen_start_time.floor('D'),
                                            sunrise_delay_in_minutes=sunrise_delay_in_minutes,
                                            sunset_delay_in_minutes=sunset_delay_in_minutes)
            
            trans_horizon_dict['sunrise'][asset] = {'timestep':trans['sunrise']['timestep'],'active':trans['sunrise']['active']}
            trans_horizon_dict['sunset'][asset] = {'timestep':trans['sunset']['timestep'],'active':trans['sunset']['active']}

        self.trans_horizon = trans_horizon_dict



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
        self.model.generate_gauss_pca_scenarios(self.trans_horizon, 
            self.hist_sun_info, nscen, upper_dict=self.meta_df.AC_capacity_MW)

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

    def pca_transform(self, num_of_components, localize=True, nearest_days=50):

        # Need to localize historical data?
        if localize:
            date_range = get_yearly_date_range(self.scen_start_time, 
                    end=(self.scen_start_time-pd.Timedelta(1,unit='D')).strftime('%Y%m%d'), num_of_days=nearest_days)
            local_days = [d+pd.Timedelta(6,unit='H') for d in date_range]

        self.old_hist_dev_df = self.hist_dev_df
        self.hist_dev_df = self.old_hist_dev_df[self.old_hist_dev_df.index.isin(local_days)]

        gpd_dict, self.gauss_df = gaussianize(self.hist_dev_df)
        self.gpd_dict = {
            (asset, timestep): gpd_dict[asset, horizon]
            for asset in self.asset_list
            for horizon, timestep in enumerate(self.scen_timesteps)
            }

        self.num_of_components = num_of_components
        self.num_of_hist_data = self.gauss_df.shape[0]

        # Center data
        mean_dict = dict()
        X_centered = np.zeros((self.num_of_assets * self.num_of_hist_data, self.num_of_horizons))

        for i,asset in enumerate(self.asset_list):
            cols = [(asset,h) for h in range(self.num_of_horizons)]
            
            X = self.gauss_df[cols].values
            
            mean_dict[asset] = dict()
            mean_dict[asset]['original'] = X
            mean_dict[asset]['mean'] = X.mean(axis=0)
            
            X_centered[i * self.num_of_hist_data:(i + 1) * self.num_of_hist_data, :] = X - X.mean(axis=0)
            
        # Fit PCA
        pca = PCA(n_components=num_of_components, svd_solver='full')
        Y = pca.fit_transform(X_centered)

        arr = np.zeros((self.num_of_hist_data, self.num_of_assets * self.num_of_components))
        for (i, asset) in enumerate(self.asset_list):
            arr[:, i * self.num_of_components:(i + 1) * self.num_of_components] = \
                Y[i * self.num_of_hist_data: (i + 1) * self.num_of_hist_data,:]
            
        self.pca_gauss_df = pd.DataFrame(data = arr, 
                columns=pd.MultiIndex.from_tuples([(asset, comp) for asset in self.asset_list
                    for comp in range(self.num_of_components)]), 
                    index=self.gauss_df.index)
        self.gauss_mean_dict = mean_dict
        self.pca = pca
        self.pca_residual = 1-pca.explained_variance_ratio_.cumsum()[-1]

        self.pca_scen_timesteps = self.scen_timesteps[0:self.num_of_components]

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
            lower_dict: Optional[pd.Series] = None,
            upper_dict: Optional[pd.Series] = None
            ) -> None:
            
        sqrt_cov = np.kron(sqrtm(self.asset_cov.values).real,
                            sqrtm(self.horizon_cov.values).real)

        # generate random draws from a normal distribution and use the model
        # parameters to transform them into normalized scenario deviations
        arr = sqrt_cov @ np.random.randn(
            len(self.asset_list) * self.num_of_components, nscen)

        pca_scen_df = pd.DataFrame(
            data=arr.T, columns=pd.MultiIndex.from_tuples(
                [(asset, horizon) for asset in self.asset_list
                 for horizon in range(self.num_of_components)]
                )
            )

        pca_scen_df.columns = pd.MultiIndex.from_tuples(
            pca_scen_df.columns).set_levels(self.pca_scen_timesteps, level=1)
        self.pca_scen_gauss_df = pca_scen_df.copy()

        # inverse pca transform
        arr = np.zeros((nscen, self.num_of_horizons * self.num_of_assets))
        for i,asset in enumerate(self.asset_list):
            cols = [(asset,t) for t in self.pca_scen_timesteps]
            arr[:, (i * self.num_of_horizons):((i+1) * self.num_of_horizons)] = \
                self.pca.inverse_transform(self.pca_scen_gauss_df[cols].values) + \
                    self.gauss_mean_dict[asset]['mean']
        scen_df = pd.DataFrame(data=arr, columns=pd.MultiIndex.from_tuples(
                    [(asset, ts) for asset in self.asset_list
                        for ts in self.scen_timesteps]
                    )
                )

        self.gauss_scen_df = scen_df.copy()

        # Fit conditional marginal distributions
        self.fit_conditional_marginal_dist(trans_timestep, hist_dict)

        # invert the Gaussian scenario deviations by the marginal distributions
        if not self.gauss:
            scen_means, scen_vars = scen_df.mean(), scen_df.std()

            # data considered as point mass if variance < 1e-2 
            scen_means[scen_vars<1e-2] = 99999.
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
