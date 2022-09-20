import pandas as pd
import numpy as np
import joblib
import os
from scipy.spatial.distance import pdist, squareform
from itertools import combinations as combns
import zipfile


# scoring function
def compute_energy_scores(scenarios: pd.DataFrame, actuals: pd.DataFrame,
                          forecasts: pd.DataFrame) -> pd.Series:
    scen_timesteps = scenarios.columns.unique(level=1)

    fcsts = forecasts.stack().loc[scen_timesteps]

    actl_devs = (actuals.stack().loc[scen_timesteps]
                 - fcsts).reorder_levels([1, 0])
    fcsts = fcsts.reorder_levels([1, 0])
    scen_devs = scenarios[fcsts.index] - fcsts

    actl_diff = (actl_devs - scen_devs).groupby(level=0, axis=1).apply(
        lambda devs: ((devs ** 2).sum(axis=1) ** 0.5).mean())
    scen_diff = scen_devs.groupby(level=0, axis=1).apply(
        lambda devs: squareform(pdist(
            devs.values, metric='euclidean')).mean()
    )
    return actl_diff - 0.5 * scen_diff


def compute_variograms(scenarios: pd.DataFrame,
                       actuals: pd.DataFrame, forecasts: pd.DataFrame,
                       order: float = 1.) -> pd.Series:
    scen_timesteps = scenarios.columns.unique(level=1)

    fcsts = forecasts.stack().loc[scen_timesteps]

    actl_devs = actuals.stack().loc[scen_timesteps] - fcsts
    scen_devs = scenarios.reorder_levels([1, 0], axis=1)[fcsts.index] - fcsts
    scores = pd.Series(0., index=scenarios.columns.unique(level=0))

    for t1, t2 in combns(actl_devs.index.unique(level=0), 2):
        actl_vars = (actl_devs.loc[t1] - actl_devs.loc[t2]).abs() ** order
        scen_vars = ((scen_devs[t1] - scen_devs[t2]).abs() ** order).mean()
        scores += (actl_vars - scen_vars) ** 2
    return scores


# process data into pgscen format
def process_data_pgscen(assets_df):
    forecasts = assets_df[assets_df['Type'] == 'Forecast']
    forecasts = forecasts.drop(columns=['Type'])
    actuals = assets_df[assets_df['Type'] == 'Actual']
    actuals = actuals.drop(columns=['Type'])
    scenarios = assets_df[assets_df['Type'] == 'Simulation']
    scenarios = scenarios.drop(columns=['Type'])

    forecasts = forecasts.set_index(['asset_id'])
    forecasts.columns.name = 'time'
    forecasts = forecasts.T

    actuals = actuals.set_index(['asset_id'])
    actuals.columns.name = 'time'
    actuals = actuals.T

    nobs = int(scenarios.shape[0])
    nassets = int(len(scenarios['asset_id'].unique()))
    nscenarios = int(nobs / nassets)

    scenarios['scen_id'] = np.tile(np.arange(nscenarios) + 1, nassets)
    scenarios = scenarios.set_index(['asset_id', 'scen_id'])
    scenarios.columns.name = 'time'
    scenarios = scenarios.stack().unstack(['asset_id', 'time'])
    return scenarios, actuals, forecasts


# List Zipfiles
os.chdir('/projects/PERFORM/Glen_Scenarios/TX_SolarWindLoad_Current/CSV/DA')
zipfiles_list = sorted(os.listdir())
zipfiles_list = [i for i in zipfiles_list if i[-3:] == 'zip']

# Check Invalid Zipfiles
invalid_zipfiles_list = []
for data_date in zipfiles_list:
    try:
        zip_ref = zipfile.ZipFile(data_date, "r")
        zip_ref.close()
    except zipfile.BadZipFile:
        invalid_zipfiles_list.append(data_date)

# Valid Zipfiles List
valid_zipfiles_list = [file for file in zipfiles_list if file not in invalid_zipfiles_list]

# Cal Scores
escores_solar = pd.DataFrame()
escores_wind = pd.DataFrame()
escores_load = pd.DataFrame()

varios_solar = pd.DataFrame()
varios_wind = pd.DataFrame()
varios_load = pd.DataFrame()

for data_date in valid_zipfiles_list:
    print('---')
    print(data_date)
    date = data_date[-12:-4]
    zip_ref = zipfile.ZipFile(data_date, "r")
    for i in zipfile.Path(zip_ref).iterdir():  # wind, solar, load
        print(i)
        assets_df = pd.DataFrame()
        for j in i.iterdir():
            df = pd.read_csv(j.open())
            df = df.drop(columns=['Index'])
            df['asset_id'] = j.name.split('.')[0]
            assets_df = pd.concat([assets_df, df], axis=0)
        scenarios, actuals, forecasts = process_data_pgscen(assets_df)
        escores = compute_energy_scores(scenarios, actuals, forecasts)
        varios = compute_variograms(scenarios, actuals, forecasts)

        if i.name == 'solar':
            escores_solar[date] = escores
            varios_solar[date] = varios

        if i.name == 'load':
            escores_load[date] = escores
            varios_load[date] = varios

        if i.name == 'wind':
            escores_wind[date] = escores
            varios_wind[date] = varios

    zip_ref.close()

# output scores files as new format
os.chdir('/projects/PERFORM/Glen_Scenarios/scores')
escores_solar.reset_index().to_csv('escores_solar_2017_2018_v090722.csv.gz', compression='gzip', index=False)
varios_solar.reset_index().to_csv('varios_solar_2017_2018_v090722.csv.gz', compression='gzip', index=False)

escores_wind.reset_index().to_csv('escores_wind_2017_2018_v090722.csv.gz', compression='gzip', index=False)
varios_wind.reset_index().to_csv('varios_wind_2017_2018_v090722.csv.gz', compression='gzip', index=False)

escores_load.reset_index().to_csv('escores_load_2017_2018_v090722.csv.gz', compression='gzip', index=False)
varios_load.reset_index().to_csv('varios_load_2017_2018_v090722.csv.gz', compression='gzip', index=False)
