import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
from pgscen.iso.ercot import create_day_ahead_load_scenario


def main():

    load_zone_actual_df = pd.read_csv('../data/Load/ERCOT/Actual/load_actual_1h_zone_2017_2018_utc.csv',
                                 parse_dates=['Time'],index_col='Time')
    load_zone_forecast_df = pd.read_csv('../data/Load/ERCOT/Day-ahead/load_day_ahead_forecast_zone_2017_2018_utc.csv',
                                    parse_dates=['Issue_time','Forecast_time'])
    load_zone_list = load_zone_actual_df.columns.tolist()

    nscen = 1000
    output_dir = '/Users/xy3134/Research/PERFORM/Data/GEMINI_scenario/test/'

    for scenario_start_time in pd.date_range(start='2018-01-02 06:00:00',periods=2,freq='D',tz='utc'):
        print(scenario_start_time)

        # Split data into historical/future
        load_zone_hist_actual_df = load_zone_actual_df[load_zone_actual_df.index<scenario_start_time]
        load_zone_hist_forecast_df = load_zone_forecast_df[load_zone_forecast_df['Forecast_time']<scenario_start_time]

        load_zone_future_actual_df = load_zone_actual_df[load_zone_actual_df.index>=scenario_start_time]
        load_zone_future_forecast_df = load_zone_forecast_df[load_zone_forecast_df['Forecast_time']>=scenario_start_time]

        md,gen = create_day_ahead_load_scenario(nscen,scenario_start_time,load_zone_list,load_zone_hist_actual_df,
            load_zone_hist_forecast_df,load_zone_future_actual_df,load_zone_future_forecast_df,output_dir,
            return_model=True,return_generator=True)

if __name__ == '__main__':
    main()