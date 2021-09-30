from pgscen.utils.solar_utils import *
from pgscen.utils.r_utils import gaussianize,gemini
from pgscen.model import gemini_model
from pgscen.generator import gemini_generator
from collections import OrderedDict
import numpy as np

class solar_engine(object):

    def __init__(self,meta_df,scenario_start_time,actual_df,forecast_df,
        forecast_resolution_in_minute=60,num_of_horizons=24,forecast_lead_time_in_hour=12):
        
        self.meta_df = meta_df.sort_values('site_ids')
        self.scenario_start_time = scenario_start_time
        self.hist_actual_df = actual_df
        self.hist_forecast_df = forecast_df
        self.forecast_resolution_in_minute = forecast_resolution_in_minute
        self.num_of_horizons = num_of_horizons
        self.forecast_lead_time_in_hour = forecast_lead_time_in_hour

        # Create empty model and generator
        asset_list = sorted(self.meta_df['site_ids'].tolist())
        self.solar_model = gemini_model(len(asset_list),asset_list,self.scenario_start_time,self.hist_actual_df,\
            self.hist_forecast_df,forecast_resolution_in_minute=self.forecast_resolution_in_minute,\
            num_of_horizons=self.num_of_horizons,forecast_lead_time_in_hour=self.forecast_lead_time_in_hour)
        self.solar_generator = gemini_generator(self.solar_model)
        

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

            gemini_md = gemini_model(len(asset_list),asset_list,scenario_start_time,self.hist_actual_df,self.hist_forecast_df,
                    forecast_resolution_in_minute=self.forecast_resolution_in_minute,num_of_horizons=num_of_horizons,
                    forecast_lead_time_in_hour=forecast_lead_time_in_hour)
            gemini_md.compute_deviation_with_horizons()
            gemini_md.compute_deviation()

            if i == 0:
                # For the base model, select historical dates whose sunrise and sunset times are within 30 min
                hist_dates = get_solar_hist_dates(self.scenario_start_time.floor('D'),meta_df,hist_start,hist_end,time_range_in_minutes=30)
            else:
                # For the sunrise/sunset model, select historical dates whose sunrise and sunset times are within 10 min
                hist_dates = get_solar_hist_dates(self.scenario_start_time.floor('D'),meta_df,hist_start,hist_end,time_range_in_minutes=10)

            # Shift historical dates by
            hour = scenario_start_time.hour
            if hour >= 6:
                gemini['hist_dates'] = [date+pd.Timedelta(hour,unit='H') for date in hist_dates]
            else:
                gemini['hist_dates'] = [date+pd.Timedelta(24+hour,unit='H') for date in hist_dates]

            gemini_md.hist_deviation_df = gemini_md.hist_deviation_df[gemini_md.hist_deviation_df.index.isin(gemini['hist_dates'])]

            # Gaussianize historical data
            gemini_md.gaussianize_hist_deviation()

            # Compute distance between assets to be used as regularization parameter
            coords = [coord for coord in zip(meta_df['longitude'].values,meta_df['latitude'].values)]
            dist = gemini_md.asset_distance(coords)

            # Normalize distance such that largest entry = 0.1.
            if np.max(dist) > 0:
                dist /= np.max(dist)*10

            # Set the distance between asset at the same location to be a small positive constanst
            # to prevent glasso from not converging
            if (dist>0).any():
                dist[dist==0] = 1e-1*np.min(dist[dist>0])
            else:
                dist += 1e-4

            gemini_md.fit(dist,0.01)

            gemini['gemini_model'] = gemini_md

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

                overlag_timesteps = list(set(gemini_gen.scen_timesteps).intersection(set(cond_gen.scen_timesteps)))
                gemini_gen_horizons = [gemini_gen.scen_timesteps.index(t) for t in overlag_timesteps]
                cond_gen_horizons = [cond_gen.scen_timesteps.index(t) for t in overlag_timesteps]

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


    def write_to_csv(self,save_dir,solar_actual_df,load_actual_df=None,forecast=True):

        if hasattr(self,'load_generator'):
            pass
            # # Load and solar joint model
            # self.load_generator.write_to_csv('load',save_dir,
            #         forecast_df=self.load_generator.forecast_df,actual_df=load_actual_df)

            # self.solar_generator.write_to_csv('solar',save_dir,
            #         forecast_df=self.solar_generator.forecast_df,actual_df=solar_actual_df)
        else:
            # Solar only
            self.solar_generator.write_to_csv('solar',save_dir,
                    forecast=forecast,actual_df=solar_actual_df)


def create_day_ahead_solar_scenario(solar_actual_df,solar_forecast_df,solar_asset_list,
        solar_meta_df,scenario_start_datetime,nscen,output_dir=None,return_engine=False):
        
    # Create a solar engine, determine parameters
    se = solar_engine(solar_meta_df,scenario_start_datetime,solar_actual_df,solar_forecast_df)
    se.get_model_params()
    se.fit_solar_model()
    se.create_solar_scenario(nscen,solar_forecast_df)
    se.write_to_csv(output_dir,solar_actual_df,forecast=True)

    if return_engine:
        return se


    # @staticmethod
    # def _fit_load_solar_joint_model(scenario_start_time,load_lag_start,load_lag_end,solar_lag_start,solar_lag_end,
    #         load_zone_list,load_deviation_df,solar_meta_df,solar_site_list,solar_deviation_df,hist_dates):
    #     """
    #     Fit joint load and solar GEMINI model
    #     """

    #     load_md = gemini_model(len(load_zone_list),
    #                 load_zone_list,
    #                 load_lag_start,
    #                 load_lag_end,
    #                 scenario_start_time)

    #     solar_md = gemini_model(len(solar_site_list),
    #                 solar_site_list,
    #                 solar_lag_start,
    #                 solar_lag_end,
    #                 scenario_start_time)

    #     # Create historical data
    #     load_md.hist_deviation_df = load_deviation_df[load_deviation_df.index.isin(hist_dates)]
    #     load_md.hist_deviation_df = load_md.hist_deviation_df[[asset+'_'+str(lag) for asset in load_zone_list for lag in range(load_lag_start,load_lag_end+1)]]

    #     solar_md.hist_deviation_df = solar_deviation_df[solar_deviation_df.index.isin(hist_dates)]
    #     solar_md.hist_deviation_df = solar_md.hist_deviation_df[[asset+'_'+str(lag) for asset in solar_site_list for lag in range(solar_lag_start,solar_lag_end+1)]]

    #     # Gaussianizze load and solar historical deviations
    #     load_md.gaussianize_hist()
    #     solar_md.gaussianize_hist()

    #     # Fit load and solar GEMINI model
    #     load_md.fit(0.01,0.01)

    #     coords = [coord for coord in zip(solar_meta_df['longitude'].values,solar_meta_df['latitude'].values)]
    #     dist = solar_md.asset_distance(coords)
    #     dist /= np.max(dist)*10
    #     solar_md.fit(dist,0.01)

    #     # Create joint model

    #     # Aggregate Gaussian level solar to zones
    #     membership = solar_meta_df.groupby('Zone')['site_ids'].apply(list).to_dict()
    #     solar_zone_gauss_df = pd.DataFrame()
    #     for zone in membership:
    #         for lag in range(solar_lag_start,solar_lag_end+1):
    #             cols = ['_'.join([site,str(lag)]) for site in membership[zone]]
    #             solar_zone_gauss_df['_'.join([zone,str(lag)])] = solar_md.gauss_df[cols].sum(axis=1)
        
    #     # Add prefix to distinguish between load and solar zones
    #     solar_zone_gauss_df = solar_zone_gauss_df.add_prefix('Solar_')
    #     solar_zone_list = [zone[:-2] for zone in solar_zone_gauss_df.columns[::solar_md.num_of_lags]]

    #     joint_md = gemini_model(len(load_zone_list)+len(solar_zone_list),load_zone_list+solar_zone_list,
    #                                 solar_lag_start,solar_lag_end,scenario_start_time)

    #     # Joint model on the Gaussian level
    #     load_solar_gauss_df = pd.concat([load_md.gauss_df[['_'.join([zone,str(lag)]) for zone in load_zone_list for lag in range(solar_lag_start,solar_lag_end+1)]],
    #                                     solar_zone_gauss_df],axis=1)
    #     joint_md.gauss_df = load_solar_gauss_df

    #     joint_md.fit(0.05,0.05)

    #     return load_md,solar_md,joint_md

    # @staticmethod
    # def _create_load_solar_joint_scenario(nscen,forecast_issue_time,solar_meta_df,load_md,solar_md,joint_md,load_forecast_df,solar_forecast_df):
        
    #     lag_start,lag_end = joint_md.lag_start,joint_md.lag_end
    #     print(lag_start,lag_end)

    #     # First generate joint load and solar scenario
    #     joint_gen = gemini_generator(joint_md)
    #     joint_gen.generate_gauss_scenario(nscen)


    #     # Separate load and solar Gaussian scenario
    #     load_zone_list = [asset for asset in joint_md.asset_list if not asset.startswith('Solar_')]
    #     print('load_zone_list',load_zone_list)
    #     load_joint_scen_df = joint_gen.gauss_scen_df[['_'.join([zone,str(lag)]) for zone in load_zone_list for lag in range(lag_start,lag_end+1)]]
        
    #     solar_zone_list = [asset for asset in joint_md.asset_list if asset.startswith('Solar_')]
    #     solar_joint_scen_df = joint_gen.gauss_scen_df[['_'.join([zone,str(lag)]) for zone in solar_zone_list for lag in range(lag_start,lag_end+1)]]

    #     # Remove ``Solar_`` prefix
    #     solar_zone_list = [zone.replace('Solar_','') for zone in solar_zone_list]
    #     solar_joint_scen_df.rename(columns={col:col.replace('Solar_','') for col in solar_joint_scen_df.columns})

    #     # Generate conditional scenario for load
    #     load_gen = gemini_generator(load_md)
    #     print(load_joint_scen_df.shape)
    #     sqrtcov,mu = load_gen.conditional_multivariate_normal_partial_time(lag_start,lag_end,load_joint_scen_df)
    #     load_gen.generate_gauss_scenario(nscen,conditional=True,sqrtcov=sqrtcov,mu=mu)
    #     load_gen.degaussianize()

    #     # Clipping to avoid negative load
    #     load_devi_min = load_md.hist_deviation_df.min().to_dict()
    #     load_gen.clip_deviations(lower_dict=load_devi_min)
    #     load_gen.add_forecast(forecast_issue_time,load_forecast_df)

    #     # Generate conditional scenario for solar
    #     membership = solar_meta_df.groupby('Zone')['site_ids'].apply(list).to_dict()
    #     solar_gen = gemini_generator(solar_md)
    #     sqrtcov,mu = solar_gen.conditional_multivariate_normal_aggregation(len(solar_zone_list),solar_zone_list,membership,solar_joint_scen_df)
    #     solar_gen.generate_gauss_scenario(nscen,conditional=True,sqrtcov=sqrtcov,mu=mu)
    #     solar_gen.degaussianize()
    #     solar_gen.add_forecast(forecast_issue_time,solar_forecast_df)
    #     solar_capacity = dict(zip(solar_meta_df['site_ids'].values,solar_meta_df['AC_capacity_MW'].values))
    #     solar_gen.clip_capacity(upper_dict=solar_capacity)

    #     return load_gen,solar_gen,joint_gen

    # def fit_joint_load_solar_model(self,load_zone_list,load_deviation_df,solar_deviation_df,hist_start='2017-01-01',hist_end='2018-12-31'):

    #     """
    #     Fit load and solar models with chosen parameters:
    #     The base model for load and solar is joint model.
        
    #     """

    #     # Localize hist_start and hist_end
    #     hist_start = pd.to_datetime(hist_start,utc=True)
    #     hist_end = pd.to_datetime(hist_end,utc=True)

    #     self.load_zones = load_zone_list
        
    #     for i in self.gemini_dict:

    #         md = self.gemini_dict[i]

    #         lag_start,lag_end = md['lag_start'],md['lag_end']
    #         asset_list = md['asset_list']
            
    #         # Extract meta data
    #         meta_df = self.meta_df[self.meta_df['site_ids'].isin(asset_list)].sort_values('site_ids')
                
    #         if i == 0:
    #             #################################### Base model ##########################################

    #             # Determine historical dates
    #             hist_dates = get_hist_dates(self.scenario_start_time.floor('D'),hist_start,hist_end,self.meta_df,range_in_seconds=1800)
    #             md['hist_dates'] = hist_dates

    #             print('number of hist data',len(hist_dates))

    #             # Fit joint load and solar GEMINI model
    #             md['load_model'],md['solar_model'],md['joint_model'] = solar_engine._fit_load_solar_joint_model(
    #                 self.scenario_start_time,self.lag_start,self.lag_end,lag_start,lag_end,load_zone_list,
    #                 load_deviation_df,meta_df,asset_list,solar_deviation_df,hist_dates)


    #         else:

    #             #################################### Conditional model ##########################################

    #             # Determine historical dates
    #             if lag_end<=self.gemini_dict[0]['lag_start']:
    #                 # Sunrise
    #                 hist_dates_sunrise = get_hist_dates(self.scenario_start_time.floor('D'),hist_start,hist_end,meta_df,sun='rise',range_in_seconds=600)
    #                 hist_dates = set(hist_dates_sunrise).intersection(set(md['conditional_model']['hist_dates']))
    #             elif lag_start>=self.gemini_dict[0]['lag_end']:
    #                 # Sunset
    #                 hist_dates_sunset = get_hist_dates(self.scenario_start_time.floor('D'),hist_start,hist_end,meta_df,sun='set',range_in_seconds=600)
    #                 hist_dates = set(hist_dates_sunset).intersection(set(md['conditional_model']['hist_dates']))
    #             else:
    #                 # Debugging
    #                 raise(RuntimeError('Expected a model outside the base model time lags, got lags from {} to {}'.format(lag_start,lag_end)))

    #             md['hist_dates'] = hist_dates

    #             # Create solar model

    #             solar_md = gemini_model(len(asset_list),asset_list,lag_start,lag_end,self.scenario_start_time)

    #             # Create historical data
    #             solar_md.hist_deviation_df = solar_deviation_df[solar_deviation_df.index.isin(hist_dates)]
    #             solar_md.hist_deviation_df = solar_md.hist_deviation_df[[site+'_'+str(lag) for site in asset_list for lag in range(lag_start,lag_end+1)]]
                
    #             solar_md.gaussianize_hist()

    #             # Replace GPDs and Gaussianized deviations by those of the conditional model
    #             for lag in range(lag_start,lag_end+1):
    #                 if lag in list(range(md['conditional_model']['solar_model'].lag_start,md['conditional_model']['solar_model'].lag_end+1)):
    #                     for asset in solar_md.asset_list:
                            
    #                         solar_md.gpd_dict[asset+'_'+str(lag)] = md['conditional_model']['solar_model'].gpd_dict[asset+'_'+str(lag)]

    #                         df = md['conditional_model']['solar_model'].gauss_df[asset+'_'+str(lag)]
    #                         df = df[df.index.isin(hist_dates)]
    #                         solar_md.gauss_df[asset+'_'+str(lag)] = df
                
    #             # Compute distance between assets to be used as regularization parameter
    #             coords = [coord for coord in zip(meta_df['longitude'].values,meta_df['latitude'].values)]
    #             dist = solar_md.asset_distance(coords)

    #             # Normalize distance such that largest entry = 0.1.
    #             if np.max(dist) > 0:
    #                 dist /= np.max(dist)*10

    #             # Set the distance between asset at the same location to be a small positive constanst
    #             # to prevent glasso from not converging
    #             if (dist>0).any():
    #                 dist[dist==0] = 1e-1*np.min(dist[dist>0])
    #             else:
    #                 dist += 1e-4
                
    #             # print('if any distance is na',dist.shape,np.isnan(dist).any())

    #             # # Check if self.gauss_df has NA
    #             # print('columns have NA',gemini_md.gauss_df.columns[gemini_md.gauss_df.isna().any()])
                
    #             # Fit graphical lasso model
    #             solar_md.fit(dist,0.01)

    #             md['solar_model'] = solar_md

    # def create_solar_scenario(self,nscen,forecast_issue_time,forecast_df):
    #     """
    #     Create scenario
    #     """

    #     capacity = dict(zip(self.meta_df['site_ids'].values,self.meta_df['AC_capacity_MW'].values))

    #     for i in self.gemini_dict:
    
    #         md = self.gemini_dict[i]
    #         gen = gemini_generator(md['gemini_model'])
            
    #         if md['conditional_model']:
    #             print(md['lag_start'],md['lag_end'])
    #             cond_gen = md['conditional_model']['gemini_generator']
                
    #             cond_gen_lags = set(range(cond_gen.lag_start,cond_gen.lag_end+1))
    #             gen_lags = set(range(gen.lag_start,gen.lag_end+1))
    #             overlap_lags = gen_lags.intersection(cond_gen_lags)
                
    #             cond_lag_start,cond_lag_end = min(overlap_lags),max(overlap_lags)
    #             cond_scen_df = cond_gen.gauss_scen_df[['_'.join([asset,str(lag)]) for asset in gen.asset_list for lag in range(cond_lag_start,cond_lag_end+1)]]
                
    #             rel_cond_lag_start = list(range(gen.lag_start,gen.lag_end+1)).index(cond_lag_start)
    #             rel_cond_lag_end = list(range(gen.lag_start,gen.lag_end+1)).index(cond_lag_end)
                
    #             print(cond_lag_start,cond_lag_end)
    #             print(rel_cond_lag_start,rel_cond_lag_end)
                
    #             sqrtcov,mu = gen.conditional_multivariate_normal_partial_time(rel_cond_lag_start,rel_cond_lag_end,cond_scen_df)
    #             gen.generate_gauss_scenario(nscen,conditional=True,sqrtcov=sqrtcov,mu=mu)
    #         else:
    #             gen.generate_gauss_scenario(nscen)
                
    #         gen.degaussianize()
    #         gen.add_forecast(forecast_issue_time,forecast_df)
            
    #         gen.clip_capacity(upper_dict=capacity)
    #         md['gemini_generator'] = gen

    #     # Collect scenario
    #     scen_df = pd.DataFrame(0.,index=[i for i in range(nscen)], \
    #                         columns=['_'.join([asset,str(lag)]) \
    #                         for asset in self.meta_df['site_ids'].tolist() \
    #                         for lag in range(self.lag_start,self.lag_end+1)])
        
    #     for i in self.gemini_dict:    
    #         gen = self.gemini_dict[i]['gemini_generator']
    #         scen_df.update(gen.scen_df)

    #     # Recreate forecast
    #     df = forecast_df[forecast_df['Issue_time']==forecast_issue_time].sort_values('Forecast_time')
    #     df = df[[asset for asset in self.meta_df['site_ids'].tolist()]]
    #     fcst_arr = np.reshape(df.values.T,(1,(self.lag_end-self.lag_start+1)*self.num_of_assets))

    #     fcst_df = pd.DataFrame(data=fcst_arr,\
    #         columns=['_'.join([asset,str(lag)]) for asset in self.meta_df['site_ids'].tolist() for lag in range(self.lag_start,self.lag_end+1)])
            
    #     gen = gemini_generator(self.gemini_dict[0]['gemini_model'])
    #     gen.scen_df = scen_df
    #     gen.forecast_df = fcst_df
    #     gen.lag_start,gen.lag_end = self.lag_start,self.lag_end

    #     self.solar_generator = gen



    # def create_load_solar_joint_scenario(self,nscen,forecast_issue_time,load_forecast_df,solar_forecast_df):
    #     """
    #     Create scenario
    #     """

    #     capacity = dict(zip(self.meta_df['site_ids'].values,self.meta_df['AC_capacity_MW'].values))

    #     for i in self.gemini_dict:
    
    #         md = self.gemini_dict[i]
            
            
    #         if md['conditional_model']:
    #             print(md['lag_start'],md['lag_end'])

    #             gen = gemini_generator(md['solar_model'])
    #             cond_gen = md['conditional_model']['solar_generator']
                
    #             cond_gen_lags = set(range(cond_gen.lag_start,cond_gen.lag_end+1))
    #             gen_lags = set(range(gen.lag_start,gen.lag_end+1))
    #             overlap_lags = gen_lags.intersection(cond_gen_lags)
                
    #             cond_lag_start,cond_lag_end = min(overlap_lags),max(overlap_lags)
    #             cond_scen_df = cond_gen.gauss_scen_df[['_'.join([asset,str(lag)]) for asset in gen.asset_list for lag in range(cond_lag_start,cond_lag_end+1)]]
                
    #             rel_cond_lag_start = list(range(gen.lag_start,gen.lag_end+1)).index(cond_lag_start)
    #             rel_cond_lag_end = list(range(gen.lag_start,gen.lag_end+1)).index(cond_lag_end)
                
    #             print(cond_lag_start,cond_lag_end)
    #             print(rel_cond_lag_start,rel_cond_lag_end)
                
    #             sqrtcov,mu = gen.conditional_multivariate_normal_partial_time(rel_cond_lag_start,rel_cond_lag_end,cond_scen_df)
    #             gen.generate_gauss_scenario(nscen,conditional=True,sqrtcov=sqrtcov,mu=mu)

    #             gen.degaussianize()
    #             gen.add_forecast(forecast_issue_time,solar_forecast_df)
                
    #             gen.clip_capacity(upper_dict=capacity)
    #             md['solar_generator'] = gen

    #         else:

    #             md['load_generator'],md['solar_generator'],md['joint_generator'] = solar_engine._create_load_solar_joint_scenario(
    #                 nscen,forecast_issue_time,self.meta_df,md['load_model'],md['solar_model'],md['joint_model'],load_forecast_df,
    #                 solar_forecast_df)                
            

    #     # Collect solar scenario
    #     scen_df = pd.DataFrame(0.,index=[i for i in range(nscen)], \
    #                         columns=['_'.join([asset,str(lag)]) \
    #                         for asset in self.meta_df['site_ids'].tolist() \
    #                         for lag in range(self.lag_start,self.lag_end+1)])
        
    #     for i in self.gemini_dict:    
    #         gen = self.gemini_dict[i]['solar_generator']
    #         scen_df.update(gen.scen_df)

    #     # Recreate forecast
    #     df = solar_forecast_df[solar_forecast_df['Issue_time']==forecast_issue_time].sort_values('Forecast_time')
    #     df = df[[asset for asset in self.meta_df['site_ids'].tolist()]]
    #     fcst_arr = np.reshape(df.values.T,(1,(self.lag_end-self.lag_start+1)*self.num_of_assets))

    #     fcst_df = pd.DataFrame(data=fcst_arr,\
    #         columns=['_'.join([asset,str(lag)]) for asset in self.meta_df['site_ids'].tolist() for lag in range(self.lag_start,self.lag_end+1)])
            
    #     # Create a gemini_generator instance to store solar scenario
    #     gen = gemini_generator(self.gemini_dict[0]['solar_model'])
    #     gen.scen_df = scen_df
    #     gen.forecast_df = fcst_df
    #     gen.lag_start,gen.lag_end = self.lag_start,self.lag_end

    #     self.solar_generator = gen

    #     # 
    #     self.load_generator = self.gemini_dict[0]['load_generator']





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