import pandas as pd
import numpy as np
from pgscen.engine import GeminiEngine
from pgscen.utils.data_utils import split_actuals_hist_future, split_forecasts_hist_future
from pathlib import Path
from pgscen.pca import PCAGeminiEngine


def load_solar_data(data_path):
    solar_site_actual_df = pd.read_csv(
        Path(data_path, 'Solar', 'NREL', 'Actual',
             'solar_actual_1h_site_2017_2018_utc.csv'),
        parse_dates=['Time'], index_col='Time'
        )

    solar_site_forecast_df = pd.read_csv(
        Path(data_path, 'Solar', 'NREL', 'Day-ahead',
             'solar_day_ahead_forecast_site_2017_2018_utc.csv'),
        parse_dates=['Issue_time', 'Forecast_time']
        )

    solar_meta_df = pd.read_excel(
        Path(data_path, 'MetaData', 'solar_meta.xlsx'))

    return solar_site_actual_df, solar_site_forecast_df, solar_meta_df


def main():

    solar_site_actual_df, solar_site_forecast_df, solar_meta_df = load_solar_data('../data/')
    nscen = 1000
    save_dir = '/Users/xy3134/Research/PERFORM/Data/Outputs/PGscen/PCA'
    


    for scen_start_time in pd.date_range(start='2018-01-02 06:00:00', periods=2, freq='D', tz='utc'):

        print(scen_start_time)
        
        scen_timesteps = pd.date_range(start=scen_start_time,periods=24, freq='H')

        (solar_site_actual_hists,
                    solar_site_actual_futures) = split_actuals_hist_future(
                            solar_site_actual_df, scen_timesteps)

        (solar_site_forecast_hists,
                    solar_site_forecast_futures) = split_forecasts_hist_future(
                            solar_site_forecast_df, scen_timesteps)

        pge = PCAGeminiEngine(solar_site_actual_hists, solar_site_forecast_hists, scen_start_time, solar_meta_df)
        dist = pge.asset_distance().values
        pge.fit(10, dist / (10 * dist.max()), 5e-2)
        pge.create_scenario(nscen, solar_site_forecast_futures)
        pge.write_to_csv(save_dir,solar_site_actual_futures,write_forecasts=True)

if __name__ == '__main__':
    main()