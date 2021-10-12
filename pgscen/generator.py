import os
import warnings
from pathlib import Path
import numpy as np
from scipy.linalg import sqrtm
import pandas as pd
from scipy.stats import norm
from pgscen.utils.r_utils import qdist,fit_dist

class gemini_generator(object):
    
    def __init__(self,model):
        """
        Initialize by copying required attributes from gemini_model
        :param md: gemini_model
        :type md: powscen.models.gemini_model.gemini_model
        """
        self.scenario_start_time = model.scenario_start_time
        self.forecast_resolution_in_minute = model.forecast_resolution_in_minute
        self.forecast_lead_time_in_hour = model.forecast_lead_time_in_hour
        self.num_of_assets = model.num_of_assets
        self.num_of_horizons = model.num_of_horizons
        self.asset_list = model.asset_list
        self.asset_cov = model.asset_cov
        self.horizon_cov = model.horizon_cov
        self.gpd_dict = model.gpd_dict

        if hasattr(model,'deviation_dict'):
            self.deviation_dict = model.deviation_dict

    
    def get_forecast(self,forecast_df):
        """
        Get forecast
        """

        self.forecast_issue_time = self.scenario_start_time-pd.Timedelta(self.forecast_lead_time_in_hour,unit='H')
        self.scenario_end_time = self.scenario_start_time+\
                pd.Timedelta((self.num_of_horizons-1)*self.forecast_resolution_in_minute,unit='min')
        self.scen_timesteps = pd.date_range(start=self.scenario_start_time,end=self.scenario_end_time,\
                freq=str(self.forecast_resolution_in_minute)+'min').strftime('%H%M').tolist()

        df = forecast_df[forecast_df['Issue_time']==self.forecast_issue_time].\
                drop(columns='Issue_time').set_index('Forecast_time')
        df = df[(df.index>=self.scenario_start_time) & (df.index<=self.scenario_end_time)].sort_index()

        self.forecast_dict = {}
        for asset in self.asset_list:
            arr = np.reshape(df[asset].values,(1,self.num_of_horizons))
            self.forecast_dict[asset] = pd.DataFrame(data=arr,columns=self.scen_timesteps)

    # def fit_conditional_gpd(self,r=0.1,positive_actual=False,minimum_sample=200):
    #     """
    #     Fit conditional GPD
    #     """
    #     self.conditional_gpd_dict = {}
    #     for asset in self.asset_list:
    #         asset_df = self.deviation_dict[asset]

    #         if positive_actual:
    #             asset_df = asset_df[asset_df['Actual']>0.]

    #         fcst_min = asset_df['Forecast'].min()
    #         capacity = asset_df['Forecast'].max()

    #         for horizon in range(self.num_of_horizons):
    #             fcst = self.forecast_dict[asset].values.ravel()[horizon]
    #             # if fcst < fcst_min or fcst > capacity:
    #             #     warnings.warn(f'forecast not in the range of historical forecasts, unable to fit a conditional GPD',RuntimeWarning)

    #             lower = max(fcst_min,fcst-r*capacity)
    #             upper = min(capacity,fcst+r*capacity)
    #             data = np.ascontiguousarray(asset_df[(asset_df['Forecast']>=lower) & (asset_df['Forecast']<=upper)]['Deviation'].values)

    #             # if len(data) < 100:
    #             #     warnings.warn(f'using {len(data)} data points to fit GPD for asset {asset} horizon {horizon}, result can be unreliable',RuntimeWarning)

    #             # If not enough samples, take more samples around forecast
    #             if len(data) < minimum_sample:
    #                 idx = (asset_df.sort_values('Forecast')-fcst).abs().sort_values('Forecast').index[0:minimum_sample]
    #                 data = np.ascontiguousarray(asset_df.loc[idx].values)

    #             self.conditional_gpd_dict['_'.join([asset,str(horizon)])] = fit_dist(data)    

    def fit_conditional_gpd(self,bin_width_ratio=0.05,positive_actual=False):
        """
        Fit conditional GPD
        """
        self.conditional_gpd_dict = {}
        for asset in self.asset_list:
            asset_df = self.deviation_dict[asset]

            if positive_actual:
                asset_df = asset_df[asset_df['Actual']>0.]

            fcst_min = asset_df['Forecast'].min()
            capacity = asset_df['Forecast'].max()

            for horizon in range(self.num_of_horizons):
                fcst = self.forecast_dict[asset].values.ravel()[horizon]

                if fcst <= 0.03*capacity:
                    # Forecast <= 3% of capacity, take 0-5% bin
                    data = np.ascontiguousarray(asset_df[asset_df['Forecast']<=0.05*capacity]['Deviation'].values)
                else:
                    # Otherwise take fcst +/- 5% bin
                    fcst_min = fcst-bin_width_ratio*capacity
                    fcst_max = fcst+bin_width_ratio*capacity
                    selected_df = asset_df[(asset_df['Forecast']>=fcst_min) & (asset_df['Forecast']<=fcst_max)]
                    data = np.ascontiguousarray(selected_df['Deviation'].values)

                if len(data) < 200:
                    warnings.warn(f'using {len(data)} data points to fit GPD for asset {asset} horizon {horizon}, result can be unreliable',RuntimeWarning)

                try:
                    self.conditional_gpd_dict['_'.join([asset,str(horizon)])] = fit_dist(data) 
                except:
                    raise RuntimeError(f'Debugging: unable to fit gpd for {asset} {horizon}')
        
    """
    Generate Gaussian scenarios
    """
    def generate_gauss_scenario(self,nscen,conditional=False,**kwargs):
        """
        Generate conditional or unconditional Gaussian scenarios. 
        :param nscen: number of scenarios
        :type nscen: int
        :param conditional: whether to generate conditional Gaussian scenarios
                            if true, use sqrtcov and mu in kwargs arguments
                            otherwise use self.asset_cov and self.horizon_cov. 
                            defaults to false
        :type conditional: boolean
        :param kwargs:
            sqrtcov: 2d numpy array, square root of the covariance matrix
            mu: 1d numpy array, mean vector
        """

        if conditional:
            sqrtcov = kwargs['sqrtcov']
            mu = kwargs['mu']
            arr = sqrtcov@np.random.randn(self.num_of_assets*self.num_of_horizons,nscen)+mu
        else:
            sqrtcov = np.kron(sqrtm(self.asset_cov.values).real,sqrtm(self.horizon_cov.values).real)
            arr = sqrtcov@np.random.randn(self.num_of_assets*self.num_of_horizons,nscen)
            
        self.gauss_scen_df = pd.DataFrame(data=arr.T,
                columns=['_'.join([asset,str(horizon)]) for asset in self.asset_list for horizon in range(self.num_of_horizons)])

    
    def conditional_multivariate_normal_partial_time(self,conditional_horizon_start,conditional_horizon_end,conditional_scen_df):
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
        # Make sure order of assets in conditional_scen_df aligned with the model
        conditional_scen_df = conditional_scen_df[['_'.join([asset,str(horizon)]) \
            for asset in self.asset_list for horizon in range(conditional_horizon_start,conditional_horizon_end+1)]] 

        R = np.zeros((self.num_of_horizons,self.num_of_horizons))
        R[conditional_horizon_start:conditional_horizon_end+1,conditional_horizon_start:conditional_horizon_end+1] = \
            np.linalg.inv(self.horizon_cov.values[conditional_horizon_start:conditional_horizon_end+1,conditional_horizon_start:conditional_horizon_end+1])
        U = np.eye(self.num_of_horizons)-self.horizon_cov.values@R
        sqrtcov = np.kron(sqrtm(self.asset_cov.values).real,sqrtm(U@self.horizon_cov.values@U.T).real)
        arr = R[:,conditional_horizon_start:conditional_horizon_end+1]
        mu = np.kron(np.eye(self.num_of_assets),self.horizon_cov.values@R[:,conditional_horizon_start:conditional_horizon_end+1])@conditional_scen_df.values.T
        return sqrtcov.real,mu

    def conditional_multivariate_normal_aggregation(self,num_of_aggregates,aggregates_list,membership,aggregates_df):

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

        S = np.zeros((num_of_aggregates,self.num_of_assets))
        for i,aggregate in enumerate(aggregates_list):
            for j,asset in enumerate(self.asset_list):
                if asset in membership[aggregate]:
                    S[i,j] = 1.
                    
        R = self.asset_cov.values@S.T@np.linalg.inv(S@self.asset_cov.values@S.T)
        mu = np.kron(R,np.eye(self.num_of_horizons))@aggregates_df.values.T
        R = np.eye(self.num_of_assets)-R@S
        sqrtcov = np.kron(sqrtm(R@self.asset_cov.values@R.T).real,sqrtm(self.horizon_cov.values).real)

        return sqrtcov.real,mu

    
    def degaussianize(self,conditional=False):
        """
        Invert the Gaussian scenarios by the marginal distributions.
        """
        self.deviation_scen_dict = {}
        for asset in self.asset_list:
            arr = np.zeros((self.gauss_scen_df.shape[0],self.num_of_horizons))
            for horizon in range(self.num_of_horizons):
                col = '_'.join([asset,str(horizon)])
                m = self.gauss_scen_df[col].mean()
                s = self.gauss_scen_df[col].std()
                u = norm.cdf((self.gauss_scen_df[col]-m)/s)

                if conditional:
                    arr[:,horizon] = qdist(self.conditional_gpd_dict[col],u)
                else:
                    arr[:,horizon] = qdist(self.gpd_dict[col],u)

            self.deviation_scen_dict[asset] = pd.DataFrame(data=arr,columns=self.scen_timesteps)

    def add_forecast(self):
        """
        Add forecast to deviation scenarios
        :param issue_time: forecast issue time
        :type issue_time: pandas Timestamp
        :param forecast_df: DataFrame stores the forecasts. 
        :type forecast_df: pandas DataFrame
                           columns of forecast_df must be
                           ``Issue_time``, ``Forecast_time`` and all asset names.
        """
        self.scen_dict = {}
        for asset in self.asset_list:
            arr = self.deviation_scen_dict[asset].values+self.forecast_dict[asset].values
            self.scen_dict[asset] = pd.DataFrame(data=arr,columns=self.scen_timesteps)

    def clip_capacity(self,lower_dict=None,upper_dict=None):
        """
        Clipping scenarios so that values outside of boundary are assigned to boundary values.
        :param lower_dict: dictionary of {asset: lower bounds}
        :type lower_dict: dict
        :param upper_dict: dictionary of {asset: upper bounds}
        :type upper_dict: dict
        """
        if lower_dict is None and upper_dict is None:
            return
        if lower_dict is None:
            lower_dict = {col:0. for col in self.asset_list}
        if upper_dict is None:
            upper_dict = {col:None for col in self.asset_list}

        for asset in self.asset_list:
            self.scen_dict[asset].clip(lower=lower_dict[asset],upper=upper_dict[asset],inplace=True)


    # def clip_deviations(self,lower_dict=None,upper_dict=None):
    #     """
    #     Clipping deviations so that values outside of boundary are assigned to boundary values.
    #     :param lower_dict: dictionary of {asset_lag: lower bounds}
    #     :type lower_dict: dict
    #     :param upper_dict: dictionary of {asset_lag: upper bounds}
    #     :type upper_dict: dict
    #     """
    #     if lower_dict is None and upper_dict is None:
    #         return
    #     if lower_dict is None:
    #         lower_dict = {col:None for col in self.deviation_scen_df.columns}
    #     if upper_dict is None:
    #         upper_dict = {col:None for col in self.deviation_scen_df.columns}

    #     for col in self.deviation_scen_df.columns:
    #         self.deviation_scen_df[col] = self.deviation_scen_df[col].clip(lower=lower_dict[col],upper=upper_dict[col])


    def write_to_csv(self,asset_type,save_dir,forecast=True,actual_df=None):
        """
        Write scenarios to CSV files
        :param asset_type: asset type, ``load``, ``wind`` or ``solar``
        :type asset_type: str
        :param save_dir: directory to save scenarios  
        :type save_dir: str of file path
        :param actual_df: actual values of the quality of interest
                          No actual included in the CSV file if actual_df=None
                          defaults to None.
        :type actual_df: None or pandas DataFrame
        :param forecast_df: forecast of the quality of interest
                            No forecast included in the CSV file if forecast_df=None
                            defaults to None.
        :type forecast_df: pandas DataFrame
        """

        scen_date = str(self.scenario_start_time.strftime('%Y%m%d'))
        
        if not os.path.exists(Path(save_dir) / scen_date):
            os.makedirs(Path(save_dir) / scen_date)

        if not os.path.exists(Path(save_dir) / scen_date / asset_type):
            os.makedirs(Path(save_dir) / scen_date / asset_type)

        csv_dir = Path(save_dir) / scen_date / asset_type
        
        for asset in self.asset_list:

            df = pd.DataFrame(columns=['Type','Index']+self.scen_timesteps)

            if actual_df is not None:
                actu_arr = actual_df[asset][(actual_df[asset].index>=self.scenario_start_time) & 
                                (actual_df[asset].index<=self.scenario_end_time)].sort_index().values
                actu_arr = np.reshape(actu_arr,(1,self.num_of_horizons))

                act_row = pd.concat([pd.DataFrame([['Actual',1]],columns=['Type','Index']),\
                        pd.DataFrame(data=actu_arr,columns=self.scen_timesteps)],axis=1)
                df = df.append(act_row)
            
            if forecast:
                fcst_row = pd.concat([pd.DataFrame([['Forecast',1]],columns=['Type','Index']),self.forecast_dict[asset]],axis=1)
                df = df.append(fcst_row)

            sim_rows = pd.concat([pd.DataFrame(data=np.concatenate(([['Simulation']]*self.scen_dict[asset].shape[0],\
                [[i] for i in range(1,self.scen_dict[asset].shape[0]+1)]),axis=1),columns=['Type','Index']),\
                self.scen_dict[asset]],axis=1)
            df = df.append(sim_rows)


            filename = asset.rstrip().replace(' ','_')+'.csv'
            df.to_csv(csv_dir / filename,index=False)