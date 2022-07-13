
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from itertools import combinations as combns


def compute_energy_scores(scenarios: pd.DataFrame, actuals: pd.DataFrame,
                          forecasts: pd.DataFrame) -> pd.Series:
    scen_timesteps = scenarios.columns.unique(level=1)

    fcsts = forecasts.set_index('Forecast_time')
    fcsts = fcsts.loc[:, ~fcsts.columns.str.contains('_time')].stack().loc[
        scen_timesteps]

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

    fcsts = forecasts.set_index('Forecast_time')
    fcsts = fcsts.loc[:, ~fcsts.columns.str.contains('_time')].stack().loc[
        scen_timesteps]

    actl_devs = actuals.stack().loc[scen_timesteps] - fcsts
    scen_devs = scenarios.reorder_levels([1, 0], axis=1)[fcsts.index] - fcsts
    scores = pd.Series(0., index=scenarios.columns.unique(level=0))

    for t1, t2 in combns(actl_devs.index.unique(level=0), 2):
        actl_vars = (actl_devs.loc[t1] - actl_devs.loc[t2]).abs() ** order
        scen_vars = ((scen_devs[t1] - scen_devs[t2]).abs() ** order).mean()
        scores += (actl_vars - scen_vars) ** 2.

    return scores
