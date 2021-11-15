"""Utility functions for loading and parsing RTS-GMLC input datasets."""

from pathlib import Path
import pandas as pd
from datetime import datetime, time


def load_load_data(input_dir):
    load_zone_actual_df = pd.read_csv(
        Path(input_dir, "RTS_Data", "timeseries_data_files",
             "Load", "REAL_TIME_regional_Load.csv")
        )[::12]

    load_zone_actual_df.Period = (load_zone_actual_df.Period - 1) // 12
    load_zone_actual_df = process_rts_actuals(parse_table_times(
        load_zone_actual_df))
    load_zone_forecast_df = pd.read_csv(
        Path(input_dir, "RTS_Data", "timeseries_data_files",
             "Load", "DAY_AHEAD_regional_Load.csv")
        )

    load_zone_forecast_df.Period -= 1
    load_zone_forecast_df = format_rts_forecasts(parse_table_times(
        load_zone_forecast_df))

    return load_zone_actual_df, load_zone_forecast_df


def load_wind_data(input_dir):
    wind_site_actual_df = pd.read_csv(Path(input_dir, "RTS_Data",
                                           "timeseries_data_files",
                                           "WIND", "REAL_TIME_wind.csv"))[::12]

    wind_site_actual_df.Period = (wind_site_actual_df.Period - 1) // 12
    wind_site_actual_df = process_rts_actuals(parse_table_times(
        wind_site_actual_df))

    wind_site_forecast_df = pd.read_csv(Path(input_dir, "RTS_Data",
                                             "timeseries_data_files",
                                             "WIND", "DAY_AHEAD_wind.csv"))

    wind_site_forecast_df.Period -= 1
    wind_site_forecast_df = format_rts_forecasts(parse_table_times(
        wind_site_forecast_df))

    gen_df = pd.read_csv(Path(input_dir, "RTS_Data", "SourceData", "gen.csv"),
                         index_col=0)
    bus_df = pd.read_csv(Path(input_dir, "RTS_Data", "SourceData", "bus.csv"),
                         index_col=0)

    wind_meta_df = pd.DataFrame({'Facility.Name': wind_site_actual_df.columns})
    wind_meta_df['Capacity'] = gen_df.loc[wind_meta_df['Facility.Name']][
        'PMax MW'].values

    gen_meta = bus_df.loc[[gen_df.loc[gen, 'Bus ID']
                           for gen in wind_meta_df['Facility.Name']]]
    wind_meta_df['lati'] = gen_meta['lat'].values
    wind_meta_df['longi'] = gen_meta['lng'].values

    return wind_site_actual_df, wind_site_forecast_df, wind_meta_df


def load_solar_data(input_dir):
    solar_site_actual_df = pd.read_csv(Path(input_dir, "RTS_Data",
                                            "timeseries_data_files",
                                            "PV", "REAL_TIME_pv.csv"))[::12]

    solar_site_actual_df.Period = (solar_site_actual_df.Period - 1) // 12
    solar_site_actual_df = process_rts_actuals(parse_table_times(
        solar_site_actual_df))

    solar_site_forecast_df = pd.read_csv(Path(input_dir, "RTS_Data",
                                             "timeseries_data_files",
                                             "PV", "DAY_AHEAD_pv.csv"))

    solar_site_forecast_df.Period -= 1
    solar_site_forecast_df = format_rts_forecasts(parse_table_times(
        solar_site_forecast_df))

    solar_roof_actual_df = pd.read_csv(Path(input_dir, "RTS_Data",
                                            "timeseries_data_files", "RTPV",
                                            "REAL_TIME_rtpv.csv"))[::12]

    solar_roof_actual_df.Period = (solar_roof_actual_df.Period - 1) // 12
    solar_roof_actual_df = process_rts_actuals(parse_table_times(
        solar_roof_actual_df))

    solar_roof_forecast_df = pd.read_csv(Path(input_dir, "RTS_Data",
                                              "timeseries_data_files", "RTPV",
                                              "DAY_AHEAD_rtpv.csv"))

    solar_roof_forecast_df.Period -= 1
    solar_roof_forecast_df = format_rts_forecasts(parse_table_times(
        solar_roof_forecast_df))

    solar_site_actual_df = solar_site_actual_df.merge(
        solar_roof_actual_df, left_index=True, right_index=True)
    solar_site_forecast_df = solar_site_forecast_df.merge(
        solar_roof_forecast_df, on=['Forecast_time', 'Issue_time'])

    gen_df = pd.read_csv(Path(input_dir, "RTS_Data", "SourceData", "gen.csv"),
                         index_col=0)
    bus_df = pd.read_csv(Path(input_dir, "RTS_Data", "SourceData", "bus.csv"),
                         index_col=0)

    solar_meta_df = pd.DataFrame({'site_ids': solar_site_actual_df.columns})
    solar_meta_df['AC_capacity_MW'] = gen_df.loc[solar_meta_df['site_ids']][
        'PMax MW'].values
    gen_meta = bus_df.loc[[gen_df.loc[gen, 'Bus ID']
                           for gen in solar_meta_df['site_ids']]]

    solar_meta_df = solar_meta_df.assign(
        latitude=gen_meta['lat'].values, longitude=gen_meta['lng'].values,
        Zone=gen_meta['Area'].astype(str).values
        )

    return solar_site_actual_df, solar_site_forecast_df, solar_meta_df


def process_rts_actuals(actual_df):
    new_df = actual_df.copy()

    new_df.index = [issue_time.tz_localize('UTC') + pd.Timedelta(hours=8)
                    for issue_time in new_df.index]
    new_df.index.name = 'Time'

    return new_df


def format_rts_forecasts(forecast_df):
    new_df = forecast_df.rename_axis('Forecast_time').reset_index()

    new_df['Issue_time'] = [
        pd.Timestamp(datetime.combine(fcst_time.date() - pd.Timedelta(days=1),
                                      time(20)),
                     tz='UTC')
        for fcst_time in new_df.Forecast_time
        ]

    new_df['Forecast_time'] = [
        fcst_time.tz_localize('UTC') + pd.Timedelta(hours=8)
        for fcst_time in new_df.Forecast_time
        ]

    return new_df


def parse_table_times(df):
    times = [datetime(year, month, day, hour)
             for year, month, day, hour in zip(df.Year, df.Month,
                                               df.Day, df.Period)]

    new_df = df.drop(columns=['Year', 'Month', 'Day', 'Period'])
    new_df.index = times

    return new_df


def get_sources_dict(input_dir):
    src_dict = dict()
    src_file = Path(input_dir, "sources_with_network.txt")

    with src_file.open() as f:
        for ln in f:
            line = ln.strip()
            eq_split = line.split('=')

            if len(eq_split) == 1:
                pr_split = line.split('(')

                if len(pr_split) == 2:
                    if pr_split[0] == 'Source':
                        src_name = pr_split[1][:-1]

                        if src_name not in src_dict:
                            src_dict[src_name] = {'files': set()}

                    else:
                        raise ValueError("Unrecognized file format!")

            elif len(eq_split) == 2:
                if eq_split[0][-5:] == '_file':
                    src_dict[src_name]['files'] |= {eq_split[1].split('"')[1]}

                else:
                    if (eq_split[0] in src_dict[src_name]
                            and (src_dict[src_name][eq_split[0]]
                                 != eq_split[1].split('"')[1])):
                        raise ValueError("Mismatching duplicate `{}` entry "
                                         "for source `{}`!".format(eq_split[0],
                                                                   src_name))

                    src_dict[src_name][eq_split[0]] = eq_split[1].split('"')[1]

    return src_dict
