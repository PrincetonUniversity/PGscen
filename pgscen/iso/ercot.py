from pgscen.utils.solar_utils import *
from pgscen.utils.r_utils import gaussianize,gemini
from pgscen.model import gemini_model
from pgscen.generator import gemini_generator
from collections import OrderedDict
import numpy as np

class solar_engine(object):

    def __init__(self,meta_df,scenario_start_time,asset_list,actual_df,forecast_df,
        forecast_resolution_in_minute=60,num_of_horizons=24,forecast_lead_time_in_hour=12):
        
        self.meta_df = meta_df.sort_values('site_ids')
        self.scenario_start_time = scenario_start_time
        self.asset_list = asset_list
        self.hist_actual_df = actual_df
        self.hist_forecast_df = forecast_df
        self.forecast_resolution_in_minute = forecast_resolution_in_minute
        self.num_of_horizons = num_of_horizons
        self.forecast_lead_time_in_hour = forecast_lead_time_in_hour

        # Create empty model and generator
        self.solar_model = gemini_model(len(asset_list),asset_list,self.scenario_start_time,self.hist_actual_df,\
            self.hist_forecast_df,forecast_resolution_in_minute=self.forecast_resolution_in_minute,\
            num_of_horizons=self.num_of_horizons,forecast_lead_time_in_hour=self.forecast_lead_time_in_hour)
        self.solar_generator = gemini_generator(self.solar_model)
        
    @staticmethod
    def get_yearly_date_range(date,num_of_days=60,start='2017-01-01',end='2018-12-31'):
        """
        Get date range around a specific date
        """
        hist_dates = pd.date_range(start=start,end=end,freq='D',tz='utc')
        hist_years = hist_dates.year.unique()
        hist_dates = set(hist_dates)

        # Take 60 days before and after
        near_dates = set()
        for year in hist_years:
            year_date = datetime(year,date.month,date.day)
            near_dates = near_dates.union(set(pd.date_range(
                start=year_date-pd.Timedelta(num_of_days,unit='D'),periods=2*num_of_days+1,freq='D',tz='utc')))
        hist_dates = hist_dates.intersection(near_dates)

        return hist_dates

    @staticmethod
    def get_solar_reg_param(asset_list,meta_df,lon_col='longitude',lat_col='latitude'):
        
        lons = meta_df.set_index('site_ids').loc[asset_list][lon_col].values
        lats = meta_df.set_index('site_ids').loc[asset_list][lat_col].values

        num_of_assets = len(asset_list)
        rho = np.zeros((num_of_assets,num_of_assets))

        for i in range(num_of_assets):
            for j in range(i+1,num_of_assets):
                rho[i,j] = np.sqrt((lons[i]-lons[j])**2+(lats[i]-lats[j])**2)
                rho[j,i] = rho[i,j]

        # Normalize distance such that largest entry = 0.1.
        if np.max(rho) > 0:
            rho /= np.max(rho)*10

        # Set the distance between asset at the same location to be a small positive constanst
        # to prevent glasso from not converging
        if (rho>0).any():
            rho[rho==0] = 1e-2*np.min(rho[rho>0])
        else:
            rho += 1e-4

        return rho

    def get_model_params(self):

        # Get local date
        local_date = self.scenario_start_time.tz_convert('US/Central').date()

        # Compute earliest and latest sunrise and sunset times
        risetimes = []
        settimes = []
        for _,row in self.meta_df.iterrows():
            
            site = row['site_ids']
            lat = row['latitude']
            lon = row['longitude']
            loc = LocationInfo(site,'Texas','USA',lat,lon)

            s = sun(loc.observer,date=local_date)
            sunrise,sunset = pd.to_datetime(s['sunrise']),pd.to_datetime(s['sunset'])

            risetimes.append(sunrise)
            settimes.append(sunset)
            
        first_sunrise,last_sunrise = min(risetimes),max(risetimes)
        first_sunset,last_sunset = min(settimes),max(settimes)

        # Get scenario timetesps
        stepsize = pd.Timedelta(self.forecast_resolution_in_minute,unit='min')
        timesteps = pd.date_range(start=self.scenario_start_time,periods=self.num_of_horizons,freq=stepsize)

        # Determine model parameters
        model_params = []
        
        horizons = []
        sunrise_period = (first_sunrise,last_sunrise)
        sunset_period = (first_sunset,last_sunset)

        for horizon,ts in enumerate(timesteps):

            # print(ts)
            
            ts_period = (ts,ts+pd.Timedelta(1,unit='H'))

            if overlap(sunrise_period,ts_period):
                # Get assets which have power production during the current time period
                active_assets = get_activate_assets('sunrise',self.meta_df,ts,\
                    time_period_in_minutes=self.forecast_resolution_in_minute,delay_in_minutes=10)

                if active_assets:
                    # How many horizons?
                    if horizon<len(timesteps)-1:
                        num_of_horizons = 2
                    else:
                        num_of_horizons = 1

                    model_params.append({'asset_list':active_assets,'scenario_start_time':ts,'num_of_horizons':num_of_horizons,\
                        'forecast_lead_time_in_hour':self.forecast_lead_time_in_hour+horizon})
                    
            elif overlap(sunset_period,ts_period):

                # Get assets which have power production during the current time period
                active_assets = get_activate_assets('sunset',self.meta_df,ts,\
                    time_period_in_minutes=self.forecast_resolution_in_minute,delay_in_minutes=10)

                if active_assets:
                    # How many horizons
                    if horizon>0:
                        num_of_horizons = 2
                    else:
                        num_of_horizons = 1

                    model_params.append({'asset_list':active_assets,'scenario_start_time':ts-stepsize,'num_of_horizons':num_of_horizons,\
                           'forecast_lead_time_in_hour':self.forecast_lead_time_in_hour+horizon-1})

            elif last_sunrise < ts < first_sunset:
                # model ts in the base model
                horizons.append(horizon)
            
        if horizons:
            model_params.append({'asset_list':sorted(self.meta_df['site_ids'].tolist()),
                    'scenario_start_time':timesteps[horizons[0]],
                    'num_of_horizons':len(horizons),
                    'forecast_lead_time_in_hour':self.forecast_lead_time_in_hour+horizons[0]})

        ######################### Determine conditional models ##################################
        
        # Sort parameters by (number of assets, number of horizons)
        model_params = sorted(model_params,key=lambda d:(len(d['asset_list']),d['num_of_horizons']),reverse=True)

        for i,param in enumerate(model_params):
            if i == 0:
                param['conditional_model'] = None
            else:
                scenario_start_time_i = param['scenario_start_time']
                scenario_end_time_i = scenario_start_time_i+(param['num_of_horizons']-1)*stepsize
                period_i = (scenario_start_time_i,scenario_end_time_i)

                for j in range(i):
                    scenario_start_time_j = model_params[j]['scenario_start_time']
                    scenario_end_time_j = scenario_start_time_j+(model_params[j]['num_of_horizons']-1)*stepsize
                    period_j = (scenario_start_time_j,scenario_end_time_j)
                    
                    if overlap(period_i,period_j):
                        param['conditional_model'] = model_params[j]
                        break
                else:
                    raise RuntimeError('Unable to find conditional model.')

        self.gemini_dict = {i:param for (i,param) in enumerate(model_params)}

    @staticmethod
    def _fit_gemini_model(asset_list,scenario_start_time,hist_actual_df,hist_forecast_df,
            forecast_resolution_in_minute,num_of_horizons,forecast_lead_time_in_hour,
            hist_deviation_index,asset_rho,horizon_rho):
        
        gemini_md = gemini_model(len(asset_list),asset_list,scenario_start_time,hist_actual_df,\
            hist_forecast_df,forecast_resolution_in_minute=forecast_resolution_in_minute,\
            num_of_horizons=num_of_horizons,forecast_lead_time_in_hour=forecast_lead_time_in_hour)
        gemini_md.compute_deviation_with_horizons()
        gemini_md.compute_deviation()
        gemini_md.hist_deviation_df = gemini_md.hist_deviation_df[gemini_md.hist_deviation_df.index.isin(hist_deviation_index)]
        gemini_md.gaussianize_hist_deviation()
        gemini_md.fit(asset_rho,horizon_rho)

        return gemini_md

    def fit_solar_model(self,hist_start='2017-01-01',hist_end='2018-12-31'):

        """
        Fit solar models with chosen parameters
        """

        for i in self.gemini_dict:

            gemini = self.gemini_dict[i]

            asset_list = gemini['asset_list']
            scenario_start_time = gemini['scenario_start_time']
            num_of_horizons = gemini['num_of_horizons']
            forecast_lead_time_in_hour = gemini['forecast_lead_time_in_hour']

            # Get meta data
            meta_df = self.meta_df[self.meta_df['site_ids'].isin(asset_list)].sort_values('site_ids')

            # Get historical data index
            if i == 0:
                # For the base model, select historical dates whose sunrise and sunset times are within 30 min
                hist_dates = get_solar_hist_dates(self.scenario_start_time.floor('D'),meta_df,\
                    hist_start,hist_end,time_range_in_minutes=30)
            else:
                # For the sunrise/sunset model, select historical dates whose sunrise and sunset times are within 10 min
                hist_dates = get_solar_hist_dates(self.scenario_start_time.floor('D'),meta_df,\
                    hist_start,hist_end,time_range_in_minutes=10)

            # Shift historical dates due to utc
            hour = scenario_start_time.hour
            if hour >= 6:
                gemini['hist_deviation_index'] = [date+pd.Timedelta(hour,unit='H') for date in hist_dates]
            else:
                gemini['hist_deviation_index'] = [date+pd.Timedelta(24+hour,unit='H') for date in hist_dates]

            # Get GEMINI regularization parameters
            asset_rho = solar_engine.get_solar_reg_param(asset_list,meta_df)
            horizon_rho = 1e-2

            gemini['gemini_model'] = solar_engine._fit_gemini_model(asset_list,scenario_start_time,\
                self.hist_actual_df,self.hist_forecast_df,self.forecast_resolution_in_minute,\
                num_of_horizons,forecast_lead_time_in_hour,gemini['hist_deviation_index'],asset_rho,horizon_rho)

    def create_solar_scenario(self,nscen,forecast_df):
        """
        Create scenario
        """

        capacity = dict(zip(self.meta_df['site_ids'].values,self.meta_df['AC_capacity_MW'].values))

        for i in self.gemini_dict:
            gemini = self.gemini_dict[i]
            gemini_md = gemini['gemini_model']
            gemini_gen = gemini_generator(gemini_md)
            gemini_gen.get_forecast(forecast_df)
            gemini_gen.fit_conditional_gpd(positive_actual=True)
            
            if gemini['conditional_model']:

                cond_gen = gemini['conditional_model']['gemini_generator']

                overlap_timesteps = list(set(gemini_gen.scen_timesteps).intersection(set(cond_gen.scen_timesteps)))
                gemini_gen_horizons = [gemini_gen.scen_timesteps.index(t) for t in overlap_timesteps]
                cond_gen_horizons = [cond_gen.scen_timesteps.index(t) for t in overlap_timesteps]

                cond_scen_df = cond_gen.gauss_scen_df[['_'.join([asset,str(horizon)]) for asset in gemini_gen.asset_list for horizon in cond_gen_horizons]]
                cond_scen_df = cond_scen_df.rename(columns={'_'.join([asset,str(j)]):'_'.join([asset,str(k)]) \
                    for asset in gemini_gen.asset_list for (j,k) in zip(cond_gen_horizons,gemini_gen_horizons)})
                
                gemini_gen_horizons = sorted(gemini_gen_horizons)
                cond_scen_df = cond_scen_df[['_'.join([asset,str(horizon)]) for asset in gemini_gen.asset_list for horizon in gemini_gen_horizons]]
                cond_horizon_start,cond_horizon_end = min(gemini_gen_horizons),max(gemini_gen_horizons)

                sqrtcov,mu = gemini_gen.conditional_multivariate_normal_partial_time(
                    cond_horizon_start,cond_horizon_end,cond_scen_df)
                gemini_gen.generate_gauss_scenario(nscen,conditional=True,sqrtcov=sqrtcov,mu=mu)
            else:
                gemini_gen.generate_gauss_scenario(nscen)

            gemini_gen.degaussianize(conditional=True)
            gemini_gen.add_forecast()
            gemini_gen.clip_capacity(upper_dict=capacity)

            gemini['gemini_generator'] = gemini_gen

        # Get forecast and collect scenarios for generator with all horizons
        self.solar_generator.get_forecast(forecast_df)
        
        # Create zero scenarios
        self.solar_generator.scen_dict = {}
        for asset in self.solar_generator.asset_list:
            self.solar_generator.scen_dict[asset] = pd.DataFrame(\
                data=np.zeros((nscen,self.solar_generator.num_of_horizons)),\
                columns=self.solar_generator.scen_timesteps)

        # Update scenarios
        for i in self.gemini_dict:
            gen = self.gemini_dict[i]['gemini_generator']
            for asset in gen.asset_list:
                df = gen.scen_dict[asset]
                self.solar_generator.scen_dict[asset].update(df)

    def import_load_data(self,load_zone_list,load_actual_df,load_forecast_df):
        self.load_zone_list = load_zone_list
        self.load_hist_actual_df = load_actual_df
        self.load_hist_forecast_df = load_forecast_df

    def _fit_load_solar_joint_model(self,load_zone_list,load_scenario_start_time,load_num_of_horizons,
            load_forecast_lead_time_in_hour,load_hist_deviation_index,solar_site_list,solar_scenario_start_time,
            solar_num_of_horizons,solar_forecast_lead_time_in_hour,solar_meta_df,solar_hist_deviation_index):
        
        # Create load model
        load_asset_rho,load_horizon_rho = 1e-2,1e-2
        load_md = solar_engine._fit_gemini_model(load_zone_list,load_scenario_start_time,\
            self.load_hist_actual_df,self.load_hist_forecast_df,self.forecast_resolution_in_minute,\
            load_num_of_horizons,load_forecast_lead_time_in_hour,load_hist_deviation_index,\
            load_asset_rho,load_horizon_rho)

        # Create solar model
        solar_asset_rho = solar_engine.get_solar_reg_param(solar_site_list,solar_meta_df)
        solar_horizon_rho = 5e-2
        solar_md = solar_engine._fit_gemini_model(solar_site_list,solar_scenario_start_time,\
            self.hist_actual_df,self.hist_forecast_df,self.forecast_resolution_in_minute,\
            solar_num_of_horizons,solar_forecast_lead_time_in_hour,solar_hist_deviation_index,\
            solar_asset_rho,solar_horizon_rho)


        ##################################### Create joint model #####################################

        # Get load data for to the same horizons in solar model
        load_gauss_df = load_md.gauss_df.copy()
        horizon_shift = int((solar_scenario_start_time-load_scenario_start_time)/\
            pd.Timedelta(self.forecast_resolution_in_minute,unit='min'))
        load_gauss_df = load_gauss_df[['_'.join([zone,str(horizon)]) for zone in load_zone_list\
             for horizon in range(horizon_shift,horizon_shift+solar_md.num_of_horizons)]]

        columns_mapping = {'_'.join([zone,str(horizon_shift+horizon)]):'_'.join([zone,str(horizon)])\
             for zone in load_zone_list for horizon in range(solar_md.num_of_horizons)}
        load_gauss_df = load_gauss_df.rename(columns=columns_mapping)
        load_gauss_df.index += horizon_shift*pd.Timedelta(self.forecast_resolution_in_minute,unit='min')


        # Get zonal solar data
        membership = solar_meta_df.groupby('Zone')['site_ids'].apply(list).to_dict()
        solar_zone_gauss_df = solar_md.aggregate_hist_gauss(membership)
        
        # Add prefix to differentiate between load and solar zones
        solar_zone_gauss_df = solar_zone_gauss_df.add_prefix('Solar_')
        solar_zone_list = [zone[:-2] for zone in solar_zone_gauss_df.columns[::solar_md.num_of_horizons]]

        # Standardize zonal data
        solar_zone_gauss_df_mean = solar_zone_gauss_df.mean()
        solar_zone_gauss_df_std = solar_zone_gauss_df.std()
        solar_zone_gauss_df = (solar_zone_gauss_df-solar_zone_gauss_df_mean)/solar_zone_gauss_df_std

        joint_md = gemini_model(len(load_zone_list+solar_zone_list),load_zone_list+solar_zone_list,\
            solar_scenario_start_time,None,None,num_of_horizons=solar_md.num_of_horizons,\
            forecast_lead_time_in_hour=solar_forecast_lead_time_in_hour)

        joint_md.solar_zone_mean = solar_zone_gauss_df_mean
        joint_md.solar_zone_std = solar_zone_gauss_df_std

        # Put load and solar zonal data together
        joint_md.gauss_df = load_gauss_df.merge(solar_zone_gauss_df,how='inner',left_index=True,right_index=True)
        joint_md.fit(0.05,0.05)

        return load_md,solar_md,joint_md

    def _create_load_solar_joint_scenario(self,nscen,meta_df,load_md,solar_md,joint_md,load_forecast_df,solar_forecast_df):

        # Generate joint load and solar scenario
        joint_gen = gemini_generator(joint_md)
        joint_gen.generate_gauss_scenario(nscen)

        # Separate load and solar Gaussian scenario
        load_zone_list = [asset for asset in joint_md.asset_list if not asset.startswith('Solar_')]
        load_joint_scen_df = joint_gen.gauss_scen_df[['_'.join([zone,str(horizon)]) \
            for zone in load_zone_list for horizon in range(joint_md.num_of_horizons)]]

        horizon_shift = int((solar_md.scenario_start_time-load_md.scenario_start_time)/\
            pd.Timedelta(self.forecast_resolution_in_minute,unit='min'))
        columns_mapping = {'_'.join([zone,str(horizon)]):'_'.join([zone,str(horizon_shift+horizon)])\
             for zone in load_zone_list for horizon in range(solar_md.num_of_horizons)}
        load_joint_scen_df = load_joint_scen_df.rename(columns=columns_mapping)
        
        solar_zone_list = [asset for asset in joint_md.asset_list if asset.startswith('Solar_')]
        solar_joint_scen_df = joint_gen.gauss_scen_df[['_'.join([zone,str(horizon)]) \
            for zone in solar_zone_list for horizon in range(joint_md.num_of_horizons)]]
        solar_joint_scen_df = solar_joint_scen_df*joint_md.solar_zone_std+joint_md.solar_zone_mean

        # Remove ``Solar_`` prefix
        solar_zone_list = [zone.replace('Solar_','') for zone in solar_zone_list]
        solar_joint_scen_df.rename(columns={col:col.replace('Solar_','') for col in solar_joint_scen_df.columns})

        # Generate conditional scenario for load
        load_gen = gemini_generator(load_md)
        load_gen.get_forecast(load_forecast_df)
        load_gen.fit_conditional_gpd()
        print(load_joint_scen_df.shape)
        cond_horizon_start = int((solar_md.scenario_start_time-load_md.scenario_start_time)\
            /pd.Timedelta(self.forecast_resolution_in_minute,unit='min'))
        cond_horizon_end = cond_horizon_start+joint_md.num_of_horizons-1
        print(cond_horizon_start,cond_horizon_end)
        sqrtcov,mu = load_gen.conditional_multivariate_normal_partial_time(cond_horizon_start,cond_horizon_end,load_joint_scen_df)
        load_gen.generate_gauss_scenario(nscen,conditional=True,sqrtcov=sqrtcov,mu=mu)
        load_gen.degaussianize(conditional=True)
        load_gen.add_forecast()

        # # Clipping to avoid negative load
        # load_devi_min = load_md.hist_deviation_df.min().to_dict()
        # load_gen.clip_deviations(lower_dict=load_devi_min)
        # load_gen.add_forecast(forecast_issue_time,load_forecast_df)

        # Generate conditional scenario for solar
        membership = self.meta_df.groupby('Zone')['site_ids'].apply(list).to_dict()
        solar_gen = gemini_generator(solar_md)
        solar_gen.get_forecast(solar_forecast_df)
        solar_gen.fit_conditional_gpd(positive_actual=True)
        sqrtcov,mu = solar_gen.conditional_multivariate_normal_aggregation(len(solar_zone_list),\
            solar_zone_list,membership,solar_joint_scen_df)
        solar_gen.generate_gauss_scenario(nscen,conditional=True,sqrtcov=sqrtcov,mu=mu)
        solar_gen.degaussianize(conditional=True)
        solar_gen.add_forecast()
        solar_capacity = dict(zip(self.meta_df['site_ids'].values,self.meta_df['AC_capacity_MW'].values))
        solar_gen.clip_capacity(upper_dict=solar_capacity)

        return load_gen,solar_gen,joint_gen

    def fit_load_solar_joint_model(self,load_zone_list,load_hist_actual_df,load_hist_forecast_df,hist_start='2017-01-01',hist_end='2018-12-31'):

        """
        Fit solar models with chosen parameters
        """

        for i in self.gemini_dict:
            
            gemini = self.gemini_dict[i]

            solar_site_list = gemini['asset_list']
            solar_scenario_start_time = gemini['scenario_start_time']
            solar_num_of_horizons = gemini['num_of_horizons']
            solar_forecast_lead_time_in_hour = gemini['forecast_lead_time_in_hour']
            solar_meta_df = self.meta_df[self.meta_df['site_ids'].isin(solar_site_list)].sort_values('site_ids')

            if i == 0:
                ##### Base model
                solar_hist_dates = get_solar_hist_dates(self.scenario_start_time.floor('D'),\
                    solar_meta_df,hist_start,hist_end,time_range_in_minutes=30)

                # Shift solar historical dates by the hour of scenario start time due to utc
                hour = solar_scenario_start_time.hour
                if hour >= 6:
                    gemini['solar_hist_deviation_index'] = [date+pd.Timedelta(hour,unit='H') for date in solar_hist_dates]
                else:
                    gemini['solar_hist_deviation_index'] = [date+pd.Timedelta(24+hour,unit='H') for date in solar_hist_dates]

                self.import_load_data(load_zone_list,load_hist_actual_df,load_hist_forecast_df)
                load_scenario_start_time = self.scenario_start_time
                load_num_of_horizons = self.num_of_horizons
                load_forecast_lead_time_in_hour = self.forecast_lead_time_in_hour
                load_hist_dates = solar_engine.get_yearly_date_range(load_scenario_start_time.floor('D'),60,hist_start,hist_end)
                
                # Shift load historical dates by the hour of scenario start time due to utc
                hour = load_scenario_start_time.hour
                if hour >= 6:
                    gemini['load_hist_deviation_index'] = [date+pd.Timedelta(hour,unit='H') for date in load_hist_dates]
                else:
                    gemini['load_hist_deviation_index'] = [date+pd.Timedelta(24+hour,unit='H') for date in load_hist_dates]

                gemini['load_model'],gemini['solar_model'],gemini['joint_model'] = self._fit_load_solar_joint_model(
                    self.load_zone_list,self.scenario_start_time,self.num_of_horizons,self.forecast_lead_time_in_hour,
                    gemini['load_hist_deviation_index'],solar_site_list,solar_scenario_start_time,solar_num_of_horizons,
                    solar_forecast_lead_time_in_hour,solar_meta_df,gemini['solar_hist_deviation_index'])
            else:   
                # Get meta data
                solar_meta_df = self.meta_df[self.meta_df['site_ids'].isin(solar_site_list)].sort_values('site_ids')

                # For the sunrise/sunset model, select historical dates whose sunrise and sunset times are within 10 min
                solar_hist_dates = get_solar_hist_dates(self.scenario_start_time.floor('D'),solar_meta_df,hist_start,hist_end,time_range_in_minutes=10)

                # Shift hours in the historical date due to utc
                hour = solar_scenario_start_time.hour
                if hour >= 6:
                    gemini['hist_deviation_index'] = [date+pd.Timedelta(hour,unit='H') for date in solar_hist_dates]
                else:
                    gemini['hist_deviation_index'] = [date+pd.Timedelta(24+hour,unit='H') for date in solar_hist_dates]

                # Get GEMINI regularization parameters
                asset_rho = solar_engine.get_solar_reg_param(solar_site_list,solar_meta_df)
                horizon_rho = 1e-2

                gemini['solar_model'] = self._fit_gemini_model(solar_site_list,solar_scenario_start_time,
                    self.hist_actual_df,self.hist_forecast_df,self.forecast_resolution_in_minute,solar_num_of_horizons,
                    solar_forecast_lead_time_in_hour,gemini['hist_deviation_index'],asset_rho,horizon_rho)

    def create_load_solar_joint_scenario(self,nscen,load_forecast_df,solar_forecast_df):
        """
        Create scenario
        """

        capacity = dict(zip(self.meta_df['site_ids'].values,self.meta_df['AC_capacity_MW'].values))

        for i in self.gemini_dict:

            gemini = self.gemini_dict[i]
            
            if gemini['conditional_model']:
                
                solar_gen = gemini_generator(gemini['solar_model'])
                solar_gen.get_forecast(solar_forecast_df)
                solar_gen.fit_conditional_gpd(positive_actual=True)

                cond_gen = gemini['conditional_model']['solar_generator']

                overlap_timesteps = list(set(solar_gen.scen_timesteps).intersection(set(cond_gen.scen_timesteps)))
                solar_gen_horizons = [solar_gen.scen_timesteps.index(t) for t in overlap_timesteps]
                cond_gen_horizons = [cond_gen.scen_timesteps.index(t) for t in overlap_timesteps]

                cond_scen_df = cond_gen.gauss_scen_df[['_'.join([asset,str(horizon)]) for asset in solar_gen.asset_list for horizon in cond_gen_horizons]]
                cond_scen_df = cond_scen_df.rename(columns={'_'.join([asset,str(j)]):'_'.join([asset,str(k)]) \
                    for asset in solar_gen.asset_list for (j,k) in zip(cond_gen_horizons,solar_gen_horizons)})
                
                solar_gen_horizons = sorted(solar_gen_horizons)
                cond_scen_df = cond_scen_df[['_'.join([asset,str(horizon)]) for asset in solar_gen.asset_list for horizon in solar_gen_horizons]]
                cond_horizon_start,cond_horizon_end = min(solar_gen_horizons),max(solar_gen_horizons)

                sqrtcov,mu = solar_gen.conditional_multivariate_normal_partial_time(
                    cond_horizon_start,cond_horizon_end,cond_scen_df)
                solar_gen.generate_gauss_scenario(nscen,conditional=True,sqrtcov=sqrtcov,mu=mu)
                solar_gen.degaussianize(conditional=True)
                solar_gen.add_forecast()
                solar_gen.clip_capacity(upper_dict=capacity)

                gemini['solar_generator'] = solar_gen
            else:
                gemini['load_generator'],gemini['solar_generator'],gemini['joint_generator'] = self._create_load_solar_joint_scenario(
                    nscen,self.meta_df,gemini['load_model'],gemini['solar_model'],gemini['joint_model'],load_forecast_df,solar_forecast_df)
        
        self.load_generator = self.gemini_dict[0]['load_generator']

        # Get forecast and collect scenarios for generator with all horizons
        self.solar_generator.get_forecast(solar_forecast_df)
        
        # Create zero scenarios
        self.solar_generator.scen_dict = {}
        for asset in self.solar_generator.asset_list:
            self.solar_generator.scen_dict[asset] = pd.DataFrame(\
                data=np.zeros((nscen,self.solar_generator.num_of_horizons)),\
                columns=self.solar_generator.scen_timesteps)

        # Update scenarios
        for i in self.gemini_dict:
            gen = self.gemini_dict[i]['solar_generator']
            for asset in gen.asset_list:
                df = gen.scen_dict[asset]
                self.solar_generator.scen_dict[asset].update(df)


    def write_to_csv(self,save_dir,load_actual_df=None,solar_actual_df=None,forecast=True):

        if hasattr(self,'load_generator'):
            # Load and solar joint model
            self.load_generator.write_to_csv('load',save_dir,
                    forecast=forecast,actual_df=load_actual_df)

            self.solar_generator.write_to_csv('solar',save_dir,
                    forecast=forecast,actual_df=solar_actual_df)
        else:
            # Solar only
            self.solar_generator.write_to_csv('solar',save_dir,
                    forecast=forecast,actual_df=solar_actual_df)


def asset_euclidean_distance(asset_list,meta_df,name_col='site_ids',
        lon_col='longitude',lat_col='latitude'):
        
    lons = meta_df.set_index(name_col).loc[asset_list][lon_col].values
    lats = meta_df.set_index(name_col).loc[asset_list][lat_col].values

    num_of_assets = len(asset_list)
    dist = np.zeros((num_of_assets,num_of_assets))

    for i in range(num_of_assets):
        for j in range(i+1,num_of_assets):
            dist[i,j] = np.sqrt((lons[i]-lons[j])**2+(lats[i]-lats[j])**2)
            dist[j,i] = dist[i,j]
    return dist

def create_day_ahead_load_scenario(nscen,scenario_start_time,load_zone_list,load_hist_actual_df,
        load_hist_forecast_df,load_future_actual_df,load_future_forecast_df,output_dir,
        return_model=True,return_generator=True):

    md = gemini_model(len(load_zone_list),load_zone_list,scenario_start_time,
        load_hist_actual_df,load_hist_forecast_df)
    md.compute_deviation()
    md.compute_deviation_with_horizons()
    md.gaussianize_hist_deviation()
    md.fit(5e-2,5e-2)

    gen = gemini_generator(md)
    gen.get_forecast(load_future_forecast_df)
    gen.fit_conditional_gpd()
    gen.generate_gauss_scenario(nscen)
    gen.degaussianize(conditional=True)
    gen.add_forecast()
    gen.write_to_csv('load',output_dir,forecast=True,actual_df=load_future_actual_df)

    if return_model and return_generator:
        return md,gen
    elif return_model:
        return md
    elif return_generator:
        return gen

def create_day_ahead_wind_scenario(nscen,scenario_start_time,wind_meta_df,wind_site_list,
        wind_hist_actual_df,wind_hist_forecast_df,wind_future_actual_df,wind_future_forecast_df,
        output_dir,return_model=False,return_generator=False):

    md = gemini_model(len(wind_site_list),wind_site_list,scenario_start_time,
        wind_hist_actual_df,wind_hist_forecast_df)
    md.compute_deviation()
    md.compute_deviation_with_horizons()
    md.gaussianize_hist_deviation()
    
    dist = asset_euclidean_distance(wind_site_list,wind_meta_df,\
        name_col='Facility.Name',lon_col='longi',lat_col='lati')
    asset_rho = dist/(10*np.max(dist))
    horizon_rho = 5e-2
    md.fit(asset_rho,horizon_rho)

    gen = gemini_generator(md)
    gen.get_forecast(wind_future_forecast_df)
    gen.fit_conditional_gpd()
    gen.generate_gauss_scenario(nscen)
    gen.degaussianize(conditional=True)
    gen.add_forecast()
    gen.write_to_csv('wind',output_dir,forecast=True,actual_df=wind_future_actual_df)

    if return_model and return_generator:
        return md,gen
    elif return_model:
        return md
    elif return_generator:
        return gen

def create_day_ahead_solar_scenario(nscen,scenario_start_datetime,solar_meta_df,solar_site_list,
        solar_hist_forecast_df,solar_hist_actual_df,solar_future_forecast_df,solar_future_actual_df=None,
        output_dir=None,return_engine=False):
        
    se = solar_engine(solar_meta_df,scenario_start_datetime,solar_site_list,
        solar_hist_actual_df,solar_hist_forecast_df)
    se.get_model_params()
    se.fit_solar_model()
    se.create_solar_scenario(nscen,solar_future_forecast_df)
    se.write_to_csv(output_dir,solar_actual_df=solar_future_actual_df,forecast=True)

    if return_engine:
        return se

def create_day_ahead_load_solar_joint_scenario(nscen,scenario_start_datetime,load_zone_list,
        load_hist_forecast_df,load_hist_actual_df,solar_meta_df,solar_site_list,
        solar_hist_forecast_df,solar_hist_actual_df,load_future_forecast_df,
        solar_future_forecast_df,load_future_actual_df=None,solar_future_actual_df=None,
        output_dir=None,return_engine=False):

    se = solar_engine(solar_meta_df,scenario_start_datetime,solar_site_list,
        solar_hist_actual_df,solar_hist_forecast_df)
    se.get_model_params()
    se.fit_load_solar_joint_model(load_zone_list,load_hist_actual_df,load_hist_forecast_df)
    se.create_load_solar_joint_scenario(nscen,load_future_forecast_df,solar_future_forecast_df)
    se.write_to_csv(output_dir,load_actual_df=load_future_actual_df,solar_actual_df=solar_future_actual_df,forecast=True)

    if return_engine:
        return se

# def create_day_ahead_load_scenario(load_deviation_df,load_forecast_df,load_asset_list,scenario_start_datetime,
#         nscen,gpd=True,load_actual_df=None,output_dir=None,return_model=False,return_generator=False):
    
#     """
#     Create day-ahead load scenarios

#     :param load_deviation_df: historical load deviations 
#     :type load_deviation_df: pandas DataFrame
    
#                              Example:

#                               |-----------------------------------------------------------------------|
#                               |                     | Coast_0 | Coast_1 | ... | East_0 | East_1 | ... |
#                               |-----------------------------------------------------------------------|
#                               | 2018-01-01 06:00:00 |         |         |     |        |        |     |
#                               |-----------------------------------------------------------------------|
                                                              
#     :param load_forecast_df: load forecast
#     :type load_forecast_df: pandas DataFrame
#     :param load_asset_list: list of load asset names
#     :type load_asset_list: list of str
#     :param scenario_start_datetime: scenario starting time
#     :type scenario_start_datetime: pandas Timestamp
#     :param nscen: number of scenarios to be created
#     :type nscen: int 
#     :param gpd: whether fit GPD
#     :type gpd: boolean
#     :param load_actual_df: load actual, not output actual if None, 
#                            defaults to None
#     :type load_actual_df: pandas DataFrame or None
#     :param output_dir: directory to output scenario CSV files
#     :type output_dir: str or file path
#     :param return_model: whether return model object
#     :type return_model: boolean
#     :param return_generator: whether return generator object
#     :type return_generator: boolean

#     :return:
#     """

#     # Fit a gemini model
#     forecast_issue_time = str(pd.to_datetime(scenario_start_datetime)-pd.Timedelta(12,unit='H'))
#     num_of_assets = len(load_asset_list)
#     lag_start = 0
#     lag_end = 23

#     md = gemini_model(num_of_assets,load_asset_list,lag_start,lag_end,scenario_start_datetime)

#     # Choose historical data and compute Gaussian copula
#     # md.hist_deviation_df = load_deviation_df[load_deviation_df.index<md.scen_start_time]
#     md.hist_deviation_df = load_deviation_df
#     md.gaussianize_hist(gpd=gpd)
#     md.fit(0.05,0.05)

#     # Create scenarios
#     gen = gemini_generator(md)
#     gen.generate_gauss_scenario(nscen)
#     gen.degaussianize()
#     gen.add_forecast(forecast_issue_time,load_forecast_df)

#     if output_dir is not None:
#         gen.write_to_csv('load',output_dir,actual_df=load_actual_df,forecast_df=gen.forecast_df)

#     if return_model and return_generator:
#         return md,gen
#     elif return_model:
#         return md,gen.scen_df
#     else:
#         return gen.scen_df

# def create_day_ahead_solar_scenario(solar_deviation_df,solar_forecast_df,solar_actual_df,
#         solar_asset_list,solar_meta_df,scenario_start_datetime,nscen,output_dir=None,return_engine=False):
        
#     # Create a solar engine, determine parameters
#     lag_start,lag_end = 0,23
#     se = solar_engine(solar_meta_df,scenario_start_datetime,lag_start,lag_end)

#     # Fit load, solar and joint models
#     se.fit_solar_model(solar_deviation_df)

#     # Create scenarios
#     forecast_issue_time = scenario_start_datetime-pd.Timedelta(12,unit='H')
#     se.create_solar_scenario(nscen,forecast_issue_time,solar_forecast_df)

#     # Write CSV files
#     se.write_to_csv(output_dir,solar_actual_df)

#     if return_engine:
#         return se

# def create_day_ahead_wind_scenario(wind_deviation_df,wind_forecast_df,wind_asset_list,wind_meta_df,
#         scenario_start_datetime,nscen,wind_actual_df=None,output_dir=None,return_model=False,return_generator=False):

#     # Fit a gemini model
#     forecast_issue_time = str(pd.to_datetime(scenario_start_datetime)-pd.Timedelta(12,unit='H'))
#     num_of_assets = len(wind_asset_list)
#     lag_start = 0
#     lag_end = 23

#     md = gemini_model(num_of_assets,wind_asset_list,lag_start,lag_end,scenario_start_datetime)

#     # Choose historical data and compute Gaussian copula
#     md.hist_deviation_df = wind_deviation_df[wind_deviation_df.index<md.scen_start_time]
#     md.gaussianize_hist()

#     coords = [coord for coord in zip(wind_meta_df['longi'].values,wind_meta_df['lati'].values)]
#     dist = md.asset_distance(coords)
#     dist /= np.max(dist)*10
#     md.fit(dist,0.01)

#     # Create scenarios
#     gen = gemini_generator(md)
#     gen.generate_gauss_scenario(nscen)
#     gen.degaussianize()
#     gen.add_forecast(forecast_issue_time,wind_forecast_df)

#     # Clip scenarios 
#     capacity = dict(zip(wind_meta_df['Facility.Name'].values,wind_meta_df['Capacity'].values))
#     gen.clip_capacity(upper_dict=capacity)

#     if output_dir is not None:
#         gen.write_to_csv('wind',output_dir,actual_df=wind_actual_df,forecast_df=gen.forecast_df)

#     if return_model and return_generator:
#         return md,gen
#     elif return_model:
#         return md,gen.scen_df
#     else:
#         return gen.scen_df
#     pass


# def create_day_ahead_load_solar_joint_scenario(solar_deviation_df,solar_forecast_df,solar_actual_df,
#         solar_asset_list,solar_meta_df,load_deviation_df,load_forecast_df,load_actual_df,load_asset_list,
#         scenario_start_datetime,nscen,output_dir=None,return_engine=False):
        
#     lag_start,lag_end = 0,23
#     se = solar_engine(solar_meta_df,scenario_start_datetime,lag_start,lag_end)

#     # Fit load, solar and joint models
#     se.fit_joint_load_solar_model(load_asset_list,load_deviation_df,solar_deviation_df)

#     # Create scenarios
#     forecast_issue_time = scenario_start_datetime-pd.Timedelta(12,unit='H')
#     se.create_load_solar_joint_scenario(nscen,forecast_issue_time,load_forecast_df,solar_forecast_df)

#     # Write CSV files
#     se.write_to_csv(output_dir,solar_actual_df,load_actual_df)

#     if return_engine:
#         return se