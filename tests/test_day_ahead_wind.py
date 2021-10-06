import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pgscen.iso.ercot import create_day_ahead_wind_scenario

def main():

    wind_site_actual_df = pd.read_csv('../data/Wind/NREL/Actual/wind_actual_1h_site_2017_2018_utc.csv',
                             parse_dates=['Time'],index_col='Time')
    wind_site_forecast_df = pd.read_csv('../data/Wind/NREL/Day-ahead/PF/wind_day_ahead_forecast_site_2018_utc.csv',
                                parse_dates=['Issue_time','Forecast_time'])
    wind_meta_df = pd.read_excel('../data/MetaData/wind_meta.xlsx')
    wind_site_list = wind_site_actual_df.columns.tolist()

    nscen = 1000
    output_dir = '/Users/xy3134/Research/PERFORM/Data/GEMINI_scenario/test/'

    for scenario_start_time in pd.date_range(start='2018-01-02 06:00:00',periods=2,freq='D',tz='utc'):
        print(scenario_start_time)

        scenario_timesteps = pd.date_range(start=scenario_start_time,periods=24,freq='H')

        wind_site_hist_actual_df = wind_site_actual_df[~wind_site_actual_df.index.isin(scenario_timesteps)]
        wind_site_hist_forecast_df = wind_site_forecast_df[~wind_site_forecast_df['Forecast_time'].isin(scenario_timesteps)]

        wind_site_future_actual_df = wind_site_actual_df[wind_site_actual_df.index.isin(scenario_timesteps)]
        wind_site_future_forecast_df = wind_site_forecast_df[wind_site_forecast_df['Forecast_time'].isin(scenario_timesteps)]

        md,gen = create_day_ahead_wind_scenario(nscen,scenario_start_time,wind_meta_df,wind_site_list,wind_site_hist_actual_df,
            wind_site_hist_forecast_df,wind_site_future_actual_df,wind_site_future_forecast_df,output_dir,
            return_model=True,return_generator=True)

if __name__ == '__main__':
    main()