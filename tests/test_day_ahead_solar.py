import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pgscen.iso.ercot import create_day_ahead_solar_scenario

def main():

    solar_meta_df = pd.read_excel('../data/MetaData/solar_meta.xlsx')
    solar_site_actual_df = pd.read_csv('../data/Solar/NREL/Actual/solar_actual_1h_site_2017_2018_utc.csv',
                                  parse_dates=['Time'],index_col='Time')
    solar_site_forecast_df = pd.read_csv('../data/Solar/NREL/Day-ahead/solar_day_ahead_forecast_site_2017_2018_utc.csv',
                                    parse_dates=['Issue_time','Forecast_time'])
    solar_site_list = solar_site_actual_df.columns.tolist()

    nscen = 1000
    output_dir = '/Users/xy3134/Research/PERFORM/Data/GEMINI_scenario/test/'

    for scenario_start_time in pd.date_range(start='2018-01-02 06:00:00',periods=2,freq='D',tz='utc'):
        print(scenario_start_time)

        solar_site_hist_actual_df = solar_site_actual_df[solar_site_actual_df.index<scenario_start_time]
        solar_site_hist_forecast_df = solar_site_forecast_df[solar_site_forecast_df['Forecast_time']<scenario_start_time]

        solar_site_future_actual_df = solar_site_actual_df[solar_site_actual_df.index>=scenario_start_time]
        solar_site_future_forecast_df = solar_site_forecast_df[solar_site_forecast_df['Forecast_time']>=scenario_start_time]

        se = create_day_ahead_solar_scenario(nscen,scenario_start_time,solar_meta_df,solar_site_list,
        solar_site_hist_forecast_df,solar_site_hist_actual_df,solar_site_future_forecast_df,
        solar_future_actual_df=solar_site_future_actual_df,output_dir=output_dir,return_engine=True)


if __name__ == '__main__':
    main()