
import sys
from pathlib import Path
import pandas as pd
from pgscen.utils.data_utils import load_wind_data


def main(argv):
    out_dir = Path(argv)
    assert out_dir.stem == 'wind', ("this test is designed for"
                                    "wind output only!")

    wind_meta_df = load_wind_data()[2].set_index('Facility.Name',
                                                 verify_integrity=True)
    wind_meta_df.index = wind_meta_df.index.str.replace('_', ' ')

    for scen_file in out_dir.glob("*.csv"):
        scen_gen = scen_file.stem.replace('_', ' ')
        out_scens = pd.read_csv(scen_file, index_col=[0, 1])

        assert out_scens.values.min() >= 0., (
            "Scenario values are negative for wind generator `{}`!".format(
                scen_gen)
            )

        assert out_scens.values.max() <= wind_meta_df.Capacity[scen_gen], (
            "Scenario values are higher than the stated capacity "
            "for wind generator `{}`!".format(scen_gen)
            )

    print("All wind scenario output tests passed.")


if __name__ == '__main__':
    assert len(sys.argv) == 2, ("an output directory must be provided as "
                                "the only argument to this script!")

    main(sys.argv[1])
