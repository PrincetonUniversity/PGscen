"""Utilities for parsing ERCOT/NREL actual and forecast datasets."""

from pathlib import Path
import pandas as pd

data_path = Path(Path(__file__).parent.parent.parent, 'data')


def split_actuals_hist_future(actual_df, scenario_timesteps, in_sample=False):
    if in_sample:
        hist_index = ~actual_df.index.isin(scenario_timesteps)
    else:
        hist_index = actual_df.index < scenario_timesteps[0]

    return actual_df[hist_index], actual_df[~hist_index]


def split_forecasts_hist_future(forecast_df, scenario_timesteps,
                                in_sample=False):
    if in_sample:
        hist_index = ~forecast_df.Forecast_time.isin(scenario_timesteps)
    else:
        hist_index = forecast_df.Forecast_time < scenario_timesteps[0]

    return forecast_df[hist_index], forecast_df[~hist_index]


################################ ERCOT ########################################
def load_load_data():
    load_zone_actual_df = pd.read_csv(
        Path(data_path, 'ERCOT', 'Load', 'Actual',
             'load_actual_1h_zone_2017_2018_utc.csv'),
        parse_dates=['Time'], index_col='Time'
        )

    load_zone_forecast_df = pd.read_csv(
        Path(data_path, 'ERCOT', 'Load', 'Day-ahead',
             'load_day_ahead_forecast_zone_2017_2018_utc.csv'),
        parse_dates=['Issue_time', 'Forecast_time']
        )

    return load_zone_actual_df, load_zone_forecast_df


def load_wind_data():
    wind_site_actual_df = pd.read_csv(
        Path(data_path, 'ERCOT', 'Wind', 'Actual',
             'wind_actual_1h_site_2017_2018_utc.csv'),
        parse_dates=['Time'], index_col='Time'
        )

    wind_site_forecast_df = pd.read_csv(
        Path(data_path, 'ERCOT', 'Wind', 'Day-ahead',
             'wind_day_ahead_forecast_site_2018_utc.csv'),
        parse_dates=['Issue_time', 'Forecast_time']
        )

    wind_meta_df = pd.read_excel(Path(data_path, 'ERCOT', 'MetaData', 'wind_meta.xlsx'))

    return wind_site_actual_df, wind_site_forecast_df, wind_meta_df


def load_solar_data():
    solar_site_actual_df = pd.read_csv(
        Path(data_path, 'ERCOT', 'Solar', 'Actual',
             'solar_actual_1h_site_2017_2018_utc.csv'),
        parse_dates=['Time'], index_col='Time'
        )

    solar_site_forecast_df = pd.read_csv(
        Path(data_path, 'ERCOT', 'Solar', 'Day-ahead',
             'solar_day_ahead_forecast_site_2017_2018_utc.csv'),
        parse_dates=['Issue_time', 'Forecast_time']
        )

    solar_meta_df = pd.read_excel(
        Path(data_path, 'ERCOT', 'MetaData', 'solar_meta.xlsx'))

    return solar_site_actual_df, solar_site_forecast_df, solar_meta_df

################################ NYISO ##########################################
def load_ny_load_data():
    load_zone_actual_df = pd.read_csv(
        Path(data_path, 'NYISO', 'Load', 'Actual',
             'load_actual_1h_zone_2018_2019_utc.csv'),
        parse_dates=['Time'], index_col='Time'
        )

    load_zone_forecast_df = pd.read_csv(
        Path(data_path, 'NYISO', 'Load', 'Day-ahead',
             'load_day_ahead_forecast_zone_2018_2019_utc.csv'),
        parse_dates=['Issue_time', 'Forecast_time']
        )

    return load_zone_actual_df, load_zone_forecast_df


def load_ny_wind_data():
    wind_site_actual_df = pd.read_csv(
        Path(data_path, 'NYISO', 'Wind', 'Actual',
             'wind_actual_1h_site_2019_utc.csv'),
        parse_dates=['Time'], index_col='Time'
        )

    wind_site_forecast_df = pd.read_csv(
        Path(data_path, 'NYISO', 'Wind', 'Day-ahead',
             'wind_day_ahead_forecast_site_2019_utc.csv'),
        parse_dates=['Issue_time', 'Forecast_time']
        )

    wind_meta_df = pd.read_csv(Path(data_path, 'NYISO', 'MetaData', 'wind_meta.csv'))

    return wind_site_actual_df, wind_site_forecast_df, wind_meta_df


def load_ny_solar_data():
    solar_site_actual_df = pd.read_csv(
        Path(data_path, 'NYISO', 'Solar', 'Actual',
             'solar_actual_1h_site_2018_2019_utc.csv'),
        parse_dates=['Time'], index_col='Time'
        )

    solar_site_forecast_df = pd.read_csv(
        Path(data_path, 'NYISO', 'Solar', 'Day-ahead',
             'solar_day_ahead_forecast_site_2018_2019_utc.csv'),
        parse_dates=['Issue_time', 'Forecast_time']
        )

    solar_meta_df = pd.read_csv(
        Path(data_path, 'NYISO', 'MetaData', 'solar_meta.csv'))

    return solar_site_actual_df, solar_site_forecast_df, solar_meta_df
