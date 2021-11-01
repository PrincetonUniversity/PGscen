
from pathlib import Path
import pandas as pd


def load_load_data(input_dir):
    load_zone_actual_df = pd.read_csv(
        Path(input_dir, 'Load', 'ERCOT', 'Actual',
             'load_actual_1h_zone_2017_2018_utc.csv'),
        parse_dates=['Time'], index_col='Time'
        )

    load_zone_forecast_df = pd.read_csv(
        Path(input_dir, 'Load', 'ERCOT', 'Day-ahead',
             'load_day_ahead_forecast_zone_2017_2018_utc.csv'),
        parse_dates=['Issue_time', 'Forecast_time']
        )

    return load_zone_actual_df, load_zone_forecast_df


def load_wind_data(input_dir):
    wind_site_actual_df = pd.read_csv(
        Path(input_dir, 'Wind', 'NREL', 'Actual',
             'wind_actual_1h_site_2017_2018_utc.csv'),
        parse_dates=['Time'], index_col='Time'
        )

    wind_site_forecast_df = pd.read_csv(
        Path(input_dir, 'Wind', 'NREL', 'Day-ahead', 'PF',
             'wind_day_ahead_forecast_site_2018_utc.csv'),
        parse_dates=['Issue_time', 'Forecast_time']
        )

    wind_meta_df = pd.read_excel(Path(input_dir, 'MetaData', 'wind_meta.xlsx'))

    return wind_site_actual_df, wind_site_forecast_df, wind_meta_df


def load_solar_data(input_dir):
    solar_site_actual_df = pd.read_csv(
        Path(input_dir, 'Solar', 'NREL', 'Actual',
             'solar_actual_1h_site_2017_2018_utc.csv'),
        parse_dates=['Time'], index_col='Time'
        )

    solar_site_forecast_df = pd.read_csv(
        Path(input_dir, 'Solar', 'NREL', 'Day-ahead',
             'solar_day_ahead_forecast_site_2017_2018_utc.csv'),
        parse_dates=['Issue_time', 'Forecast_time']
        )

    solar_meta_df = pd.read_excel(
        Path(input_dir, 'MetaData', 'solar_meta.xlsx'))

    return solar_site_actual_df, solar_site_forecast_df, solar_meta_df


def split_actuals_hist_future(actual_df, scen_start_time):
    hist_df = actual_df[actual_df.index < scen_start_time]
    future_df = actual_df[actual_df.index >= scen_start_time]

    return hist_df, future_df


def split_forecasts_hist_future(forecast_df, scen_start_time):
    hist_df = forecast_df[forecast_df.Forecast_time < scen_start_time]
    future_df = forecast_df[forecast_df.Forecast_time >= scen_start_time]

    return hist_df, future_df


def split_actuals_hist_future_wind(actual_df, scenario_timesteps):
    hist_df = actual_df[~actual_df.index.isin(scenario_timesteps)]
    future_df = actual_df[actual_df.index.isin(scenario_timesteps)]

    return hist_df, future_df


def split_forecasts_hist_future_wind(forecast_df, scenario_timesteps):
    hist_df = forecast_df[~forecast_df.Forecast_time.isin(scenario_timesteps)]
    future_df = forecast_df[forecast_df.Forecast_time.isin(scenario_timesteps)]

    return hist_df, future_df
