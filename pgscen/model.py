import warnings
import numpy as np
import pandas as pd
from pgscen.utils.r_utils import gaussianize,graphical_lasso,gemini

class gemini_model(object):

    def __init__(self,num_of_assets,asset_list,scenario_start_time,actual_df,forecast_df,
            forecast_resolution_in_minute=60,num_of_horizons=24,forecast_lead_time_in_hour=12):
        """
        Initialization.

        :param num_of_assets: number of assets
        :type num_of_assets: int
        :param asset_list: list of asset names
        :type asset_list: list of str
        :param lag_start: first lag
        :type lag_start: int
        :param lag_end: last lag
        :type lag_end: int
        :param scen_start_time: time when scenario starts
        :type scen_start_time: pandas Timestamp
        :param forecast_resolution_in_minute: forecast time resolution in minutes, defaults to 60 (hourly forecast)
        :type forecast_resolution_in_minute: int
        """
        self.num_of_assets = num_of_assets
        self.asset_list = asset_list
        self.scenario_start_time = scenario_start_time
        self.hist_actual_df = actual_df
        self.hist_forecast_df = forecast_df
        self.forecast_resolution_in_minute = forecast_resolution_in_minute
        self.num_of_horizons = num_of_horizons
        self.forecast_lead_time_in_hour = forecast_lead_time_in_hour

        self.asset_cov = None
        self.horizon_cov = None
        self.gpd_dict = None

    def compute_deviation(self):
        """
        Put actual, forecast and deviation into one pandas DataFrame
        """
        df = self.hist_actual_df.merge(self.hist_forecast_df,how='inner',left_on='Time',\
            right_on='Forecast_time',suffixes=('_actual','_forecast'))

        self.deviation_dict = {}
        for asset in self.asset_list:
            act = df[asset+'_actual']
            fcst = df[asset+'_forecast']
            self.deviation_dict[asset] = pd.DataFrame({'Actual':act,'Forecast':fcst,'Deviation':act-fcst})

    def compute_deviation_with_horizons(self):
        """
        Compute deviation from historical data: deviation = actual - forecast.
        """
        asset_lag_columns = ['_'.join([asset,str(lag)]) for asset in self.asset_list for lag in range(self.num_of_horizons)]
        self.hist_deviation_df = pd.DataFrame(columns=asset_lag_columns)

        gb = self.hist_forecast_df.groupby('Issue_time')
        for issue_time in gb.groups.keys():
            forecast_start_time = issue_time+pd.Timedelta(self.forecast_lead_time_in_hour,unit='H')
            forecast_end_time = forecast_start_time+\
                pd.Timedelta(self.forecast_resolution_in_minute*(self.num_of_horizons-1),unit='min')

            # print(issue_time,forecast_start_time,forecast_end_time)

            # Get actual
            act_df = self.hist_actual_df[(self.hist_actual_df.index>=forecast_start_time) & (self.hist_actual_df.index<=forecast_end_time)][self.asset_list]

            # Get forecast
            fcst_df = gb.get_group(issue_time).drop(columns='Issue_time').set_index('Forecast_time').sort_index()
            fcst_df = fcst_df[(fcst_df.index>=forecast_start_time) & (fcst_df.index<=forecast_end_time)][self.asset_list]
            
            # Create lagged deviations
            if act_df.shape != (self.num_of_horizons,self.num_of_assets):
                warnings.warn(f'unable to find actual data to be matched with forecast issued at {issue_time}',RuntimeWarning)
            elif fcst_df.shape != (self.num_of_horizons,self.num_of_assets):
                warnings.warn(f'forecast issued at {issue_time} does not have {self.num_of_horizons} horizons',RuntimeWarning)
            else:
                # Compute difference
                arr = np.reshape(act_df.values.T,(1,self.num_of_assets*self.num_of_horizons))-\
                    np.reshape(fcst_df.values.T,(1,self.num_of_assets*self.num_of_horizons))
            
                self.hist_deviation_df = self.hist_deviation_df.append(pd.DataFrame(data=arr,
                    index=[forecast_start_time],columns=asset_lag_columns))

    def aggregate_hist_gauss(self,membership):
        aggregate_df = pd.DataFrame()
        for aggregate in membership:
            for horizon in range(self.num_of_horizons):
                cols = ['_'.join([asset,str(horizon)]) for asset in membership[aggregate]]
                aggregate_df['_'.join([aggregate,str(horizon)])] = self.gauss_df[cols].sum(axis=1)
        return aggregate_df

    # def get_forecast_deviation_pair(self):
    #     """
    #     Get forecast/deviation pair
    #     """
    #     df = self.hist_actual_df.merge(self.hist_forecast_df,how='inner',left_on='Time',\
    #         right_on='Forecast_time',suffixes=('_actual','_forecast'))

    #     self.forecast_deviation_pair_dict = {}
    #     for asset in self.asset_list:
    #         act = df[asset+'_actual']
    #         fcst = df[asset+'_forecast']
    #         self.forecast_deviation_pair_dict[asset] = pd.DataFrame({'Forecast':fcst,'Deviation':act-fcst})

    def gaussianize_hist_deviation(self):
        """
        Make historical data ``Gaussian``

        :param gpd: wether to fit GPD
        :type gpd: boolean
        :param gauss: whether the data is already Gaussian
        :type gauss: boolean
        :param trend: whether to fit a trend (currently not supported)
        :type trend: boolean
        """
        self.gpd_dict,self.gauss_df = gaussianize(self.hist_deviation_df)

    def fit(self,asset_rho,horizon_rho):
        """
        Fit graphical lasso or gemini model

        :param asset_rho: regularization parameter for spatial (asset) dimension
        :type asset_rho: float or (n,n) numpy array, where n is number of assets
        :param lag_rho: regularization parameter for temporal (lag) dimension
        :type lag_rho: float or (n,n) numpy array, where n is number of lags

        """
        if self.num_of_assets==1:
            # Only one asset run plain GLASSO
            horizon_prec = graphical_lasso(self.gauss_df,self.num_of_horizons,horizon_rho)
            asset_prec = np.array([[1.0]])
        elif self.num_of_horizons==1:
            # Only one horizon run plain GLASSO
            asset_prec = graphical_lasso(self.gauss_df,self.num_of_assets,asset_rho)
            horizon_prec = np.array([[1.0]])
        else:
            # Multiple assets and lags, run GEMINI
            asset_prec,horizon_prec = gemini(self.gauss_df,self.num_of_assets,self.num_of_horizons,asset_rho,horizon_rho)
        
        # Compute covariance matrices
        asset_cov = np.linalg.inv(asset_prec)
        self.asset_cov = pd.DataFrame(data=(asset_cov+asset_cov.T)/2,
            index=self.asset_list,columns=self.asset_list)

        horizon_cov = np.linalg.inv(horizon_prec)
        self.horizon_cov = pd.DataFrame(data=(horizon_cov+horizon_cov.T)/2,
            index=['_'.join(['horizon',str(horizon)]) for horizon in range(self.num_of_horizons)],
            columns=['_'.join(['horizon',str(horizon)]) for horizon in range(self.num_of_horizons)])

    def asset_distance(self,coords):
        """
        Compute distance between a set of locations

        :param coords: list of coordinates, the order of the list must match self.asset_list
        :type coords: list of coordinates (float,float)
        
        :return: 2d numpy array stores the distance
        """
        dist = np.zeros((len(self.asset_list),len(self.asset_list)))

        for i in range(self.num_of_assets):
            for j in range(i+1,self.num_of_assets):
                dist[i,j] = np.sqrt((coords[i][0]-coords[j][0])**2+(coords[i][1]-coords[j][1])**2)
                dist[j,i] = dist[i,j]
        return dist