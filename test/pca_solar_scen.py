import pandas as pd
import numpy as np
from pgscen.engine import GeminiEngine
from pgscen.utils.data_utils import split_actuals_hist_future, split_forecasts_hist_future
from pathlib import Path
from pgscen.pca import PCAGeminiEngine
from pgscen.utils.solar_utils import get_yearly_date_range


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
    # save_dir = '/Users/xy3134/Research/PERFORM/Data/Outputs/PGscen/PCA'
    save_dir = '/projects/PERFORM/xyang/scenarios/pca2'


    for scen_start_time in pd.date_range(start='2018-12-17 06:00:00', periods=1, freq='D', tz='utc'):

        print(scen_start_time)
        
        scen_timesteps = pd.date_range(start=scen_start_time,periods=24, freq='H')


         # Get historical data

        (solar_site_actual_hists,
                    solar_site_actual_futures) = split_actuals_hist_future(
                            solar_site_actual_df, scen_timesteps)

        (solar_site_forecast_hists,
                    solar_site_forecast_futures) = split_forecasts_hist_future(
                            solar_site_forecast_df, scen_timesteps)

        hist_dates = sorted(get_yearly_date_range(date=scen_start_time,num_of_days=50,
                      start=str(solar_site_actual_hists.index.min().date()),
                      end=str(solar_site_actual_hists.index.max().date())))[:-1]
        hist_fcst_issue_times = [t-pd.Timedelta(6,unit='H') for t in hist_dates]

        solar_site_forecast_hists = solar_site_forecast_hists[solar_site_forecast_hists['Issue_time'].isin(hist_fcst_issue_times)]
        hist_start = solar_site_forecast_hists['Forecast_time'].min()
        hist_end = solar_site_forecast_hists['Forecast_time'].max()
        solar_site_actual_hists = solar_site_actual_hists[(solar_site_actual_hists.index>=hist_start) & \
                                                        (solar_site_actual_hists.index<=hist_end)]

        # Fit model and generate scenarios
        pge = PCAGeminiEngine(solar_site_actual_hists, solar_site_forecast_hists, scen_start_time, solar_meta_df)
        dist = pge.asset_distance().values
        pge.fit(10, dist / (10 * dist.max()), 5e-2)
        pge.create_scenario(nscen, solar_site_forecast_futures)
        pge.write_to_csv(save_dir,solar_site_actual_futures,write_forecasts=True)

        print(pge.model.hist_dev_df.shape)

if __name__ == '__main__':
    main()
