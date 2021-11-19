# PGscen #

This package generates power grid scenarios using GEMINI trained on historical grid asset actual and forecasted values.


## Installation ##

To install PGscen, first clone this repository at the latest release:

```git clone https://github.com/PrincetonUniversity/PGscen.git --branch v0.2.0-a0 --single-branch```

Then navigate to the cloned directory and run:

```pip install .```


## Running PGscen on Taxas 7k ##

Once installed, you can generate scenarios using the NREL/ERCOT datasets for the Texas 7k system installed as part of
the package. This is easiest done using the command line interface made available upon installation.

In particular, this package includes four command line tools: `pgscen-load`, `pgscen-wind`, `pgscen-solar`, and
`pgscen-load-solar`, each of which create scenarios for the given asset type. While the first three consider load, wind,
and solar assets separately, `pgscen-load-solar` models load and solar assets jointly to account for the fact that both
are influenced by the same factors, and especially the weather.

The interface for all four of these tools is the same, with two required arguments:
 - `start`: The starting date for scenario generation, given in YYYY-MM-DD format. Currently supported dates are
            2017-01-03 to 2018-12-30 for load, load-solar, and solar scenarios, and 2018-01-01 to 2018-12-30 for wind
            scenarios.
 - `days`: The number of days to create scenarios for. Note that the implied date range must also fall in the intervals
           listed above for `start`.

These tools also take the following optional arguments:
 - `--out-dir` (`-o`): Where to store generated scenarios. By default, output files are stored in the `PGscen/data`
                       subdirectory of where this repository was cloned.
 - `--scenario-count` (`-n`): How many scenarios to generate. By default, 1000 are created.
 - `--verbose`: Whether to print messages on the state of the scenario generator. Use `-v`, `-vv` for escalating levels
                of verbosity.
 - `--test`: Run scenario generation using cached subsets of the ERCOT/NREL datasets to test if installation was carried
             out correctly.

Example usages include:

`pgscen-load-solar 2017-06-29 2`

`pgscen-load-solar 2018-01-25 2 -v`

`pgscen-load 2018-03-02 2 -o scratch/output/PGscen`

`pgscen-wind 2018-07-05 3 -n 2000`

`pgscen-solar 2017-09-22 1 -n 500 -vv`
