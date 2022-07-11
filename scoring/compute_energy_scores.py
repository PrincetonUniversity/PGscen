
import argparse
from pathlib import Path
import bz2
import dill as pickle
from scipy.spatial.distance import pdist, squareform

from pgscen.rts_gmlc.data_utils import load_load_data as load_rts_load_data
from pgscen.rts_gmlc.data_utils import load_wind_data as load_rts_wind_data
from pgscen.rts_gmlc.data_utils import load_solar_data as load_rts_solar_data


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("scen_file", type=Path,
                        help="a pickle file with PGscen output for one day")
    parser.add_argument("rts_dir", type=Path)

    args = parser.parse_args()
    Path(args.scen_file.parent, "scores").mkdir(exist_ok=True)

    load_zone_actual_df, load_zone_forecast_df = load_rts_load_data(
        args.rts_dir)
    (wind_site_actual_df, wind_site_forecast_df,
        wind_meta_df) = load_rts_wind_data(args.rts_dir)
    (solar_site_actual_df, solar_site_forecast_df,
        solar_meta_df) = load_rts_solar_data(args.rts_dir)

    load_zone_forecast_df.set_index('Forecast_time', inplace=True)
    wind_site_forecast_df.set_index('Forecast_time', inplace=True)
    solar_site_forecast_df.set_index('Forecast_time', inplace=True)

    forecast_dfs = {
        'Load': load_zone_forecast_df.loc[
            :, ~load_zone_forecast_df.columns.str.contains('_time')].stack(),
        'Wind': wind_site_forecast_df.loc[
            :, ~wind_site_forecast_df.columns.str.contains('_time')].stack(),
        'Solar': solar_site_forecast_df.loc[
            :, ~solar_site_forecast_df.columns.str.contains('_time')].stack()
        }

    load_zone_actual_df = load_zone_actual_df.stack()
    wind_site_actual_df = wind_site_actual_df.stack()
    solar_site_actual_df = solar_site_actual_df.stack()

    actl_devs = {'Load': load_zone_actual_df - forecast_dfs['Load'],
                 'Wind': wind_site_actual_df - forecast_dfs['Wind'],
                 'Solar': solar_site_actual_df - forecast_dfs['Solar']}

    scores = dict()
    with bz2.BZ2File(args.scen_file, 'r') as f:
        day_scens = pickle.load(f)

    scen_timesteps = {tuple(scen_df.columns.get_level_values(1).unique())
                      for scen_df in day_scens.values()}
    assert len(scen_timesteps) == 1
    scen_timesteps = list(tuple(scen_timesteps)[0])

    for coh, coh_scens in day_scens.items():
        actl_dev = actl_devs[coh].loc[scen_timesteps].reorder_levels([1, 0])
        fcst_vals = forecast_dfs[coh].loc[scen_timesteps].reorder_levels(
            [1, 0])
        scen_devs = coh_scens[fcst_vals.index] - fcst_vals

        actl_diff = (actl_dev - scen_devs).groupby(level=0, axis=1).apply(
            lambda devs: ((devs ** 2).sum(axis=1) ** 0.5).mean())
        scen_diff = scen_devs.groupby(level=0, axis=1).apply(
            lambda devs: squareform(pdist(
                devs.values, metric='euclidean')).mean()
            )

        scores[coh] = actl_diff - 0.5 * scen_diff

    out_fl = Path(args.scen_file.parent, "scores", args.scen_file.stem)
    with open(out_fl, 'wb') as f:
        pickle.dump(scores, f, protocol=-1)


if __name__ == '__main__':
    main()
