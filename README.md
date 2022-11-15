[![CI](https://github.com/PrincetonUniversity/PGscen/actions/workflows/main.yml/badge.svg)](https://github.com/PrincetonUniversity/PGscen/actions/workflows/main.yml)

# PGscen #

This package generates power grid scenarios using GEMINI trained on historical grid asset actual and forecasted values.
Please see the Jupyter notebooks available in the `examples/` directory for an overview of how PGscen works.


## Installation ##

PGscen is available for installation on Linux-based operating systems such as Ubuntu as well as macOS. To install
PGscen, first clone this repository at the latest release:

```git clone https://github.com/PrincetonUniversity/PGscen.git --branch v0.2.0-rc.5 --single-branch```

Next, navigate to the cloned directory to create and activate the conda environment containing the prerequisite
packages for PGscen:

```conda env create -f environment.yml```

```conda activate pgscen```

From within the same directory, complete installation of PGscen by running:

```pip install .```


## Running PGscen on Texas 7k ##

Once installed, you can generate scenarios using the NREL/ERCOT datasets for the Texas 7k system installed as part of
the package. This is easiest done using the command line interface made available upon installation.

In particular, this package includes four command line tools: `pgscen-load`, `pgscen-wind`, `pgscen-solar`, and
`pgscen-load-solar`, each of which create scenarios for the given asset type. While the first three consider load, wind,
and solar assets separately, `pgscen-load-solar` models load and solar assets jointly to account for the fact that both
are influenced by the same factors, and especially the weather. You can also use the `pgscen` command, which generates
all three of load, wind, and solar scenarios at the same time.

The basic interface for all five of these tools is the same, with two required arguments:
 - `start`: The starting date for scenario generation, given in YYYY-MM-DD format. Currently supported dates (all 
            inclusive) are:
   - 2017-01-03 to 2018-12-31 for load scenarios
   - 2018-01-01 to 2018-12-30 for wind scenarios
   - 2017-01-04 to 2018-12-30 for solar scenarios, except for 2017-12-31 and 2018-01-01
   - 2017-01-05 to 2018-12-30 for load-solar scenarios, except for 2017-12-31 and 2018-01-01
 - `days`: The number of days to create scenarios for. Note that the implied date range must also fall within the
           intervals listed above for `start`.

Unlike the other tools, `pgscen` takes the optional `--joint` argument, in which case it will generate load and solar
scenarios jointly instead of the default behaviour of modeling them separately.

These tools also all take the following optional arguments:
 - `--out-dir` (`-o`): Where to store generated scenarios. By default, output files are stored in the `PGscen/data`
                       subdirectory of where this repository was cloned.
 - `--scenario-count` (`-n`): How many scenarios to generate. By default, 1000 are created.
 - `--verbose`: Whether to print messages on the state of the scenario generator. Use `-v`, `-vv` for escalating levels
                of verbosity.
 - `--pickle` (`-p`): Instead of writing a separate .csv file for each asset's daily scenarios, save the generated
                      scenarios for all assets for each day as a single Python pickle file
 - `--test`: Run scenario generation using cached subsets of the ERCOT/NREL datasets to test if installation was carried
             out correctly.

Example usages include:

`pgscen-load-solar 2017-06-29 2`

`pgscen-load-solar 2018-01-25 2 -p -v`

`pgscen-load 2018-03-02 2 -o scratch/output/PGscen`

`pgscen-wind 2018-07-05 3 -n 2000`

`pgscen 2018-05-11 3 -p`

`pgscen-solar 2017-09-22 1 -n 500 -vv`

`pgscen 2018-05-11 3 --joint -o scratch/output/PGscen -p`

The implementations of these command line interfaces are located at `pgscen/command_line.py`, and can be used as
templates for designing your own PGscen workflows. See for example `pgscen/rts_gmlc/command_line.py` for one such custom
implementation which is described in greater detail below.


## Running PGscen on RTS-GMLC ##

PGscen also includes a beta version of the `pgscen-rts` command line tool for generating scenarios for the RTS-GMLC
power grid system. The data for this system is not included with the package but can be acquired by cloning the RTS-GMLC
repository:

```git clone https://github.com/GridMod/RTS-GMLC.git```

The interface for `pgscen-rts` is identical to that of its Texas 7k counterparts, with the exception of having to
specify the directory where the RTS-GMLC dataset is located. For example, if you have a directory called `repositories`
within which you cloned the RTS-GMLC repo, you would run:

`pgscen-rts 2020-04-03 4 repositories/RTS-GMLC`

Note that this tool creates both joint load-solar and wind scenarios in one run; it also saves output as a single
compressed Python pickle object for each day unless the `--csv` flag is given, in which case it reverts to the original
format used by the Texas 7k tools. `pgscen-rts` supports any day from 2020-01-01 to 2020-12-31 inclusive.

Please see `pgscen/rts_gmlc/create_scenarios.sh` for an example of how this tool can be used to generate RTS-GMLC
scenarios for the entire year using a Slurm compute cluster.
