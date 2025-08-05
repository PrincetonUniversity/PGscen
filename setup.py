#!/bin/usr/env python
import os
import sys
import setuptools
from pathlib import Path


setuptools.setup(
    name='pgscen',
    version='0.2.1-a2',
    description="power grid scenario creation platform for "
                "load and production of wind and solar",

    author='Xinshuo Yang, Michal Grzadkowski, René Carmona',
    author_email='xy3134@princeton.edu, mgrzad@princeton.edu, '
                 'rcarmona@princeton.edu',

    packages=['pgscen', 'pgscen.utils', 'pgscen.rts_gmlc'],
    package_data={'pgscen': ['../test/resources/*.p.gz',
                             '../data/*/*/*/*.csv',
                             '../data/*/*/*/*/*.csv',
                             '../data/*/*/*.xlsx',
                             '../data/*/*/*.zip',
                             '../data/*/*/*.csv',
                             '../data/Map/*.zip']},

    entry_points={
        'console_scripts': [
            'pgscen-load=pgscen.command_line:create_load_scenarios',
            'pgscen-wind=pgscen.command_line:create_wind_scenarios',
            'pgscen-solar=pgscen.command_line:create_solar_scenarios',
            'pgscen-load-solar'
            '=pgscen.command_line:create_load_solar_scenarios',
            'pgscen=pgscen.command_line:create_scenarios',

            'pgscen-ny-load=pgscen.command_line:create_ny_load_scenarios',
            'pgscen-ny-wind=pgscen.command_line:create_ny_wind_scenarios',
            'pgscen-ny-solar=pgscen.command_line:create_ny_solar_scenarios',
            'pgscen-ny=pgscen.command_line:create_ny_scenarios',

            'pgscen-rts=pgscen.rts_gmlc.command_line:create_scenarios',
            ],
        },

# Move more installation requirements to env file, identify versions.

    install_requires=['numpy', 'matplotlib', 'scipy',
                      'dill', 'statsmodels', 'cffi', 'jupyterlab',
                      'seaborn', 'openpyxl', 'geopandas',
                      'scikit-learn', 'ipywidgets', 'astral'],
    )

# Manual installation of Rsafd dependencies.

os.system('PKG_CPPFLAGS="-DHAVE_WORKING_LOG1P" Rscript -e "install.packages(c(\'timeDate\', \'quadprog\', \'quantreg\', \'plot3D\', \'robustbase\', \'scatterplot3d\', \'splines\', \'tseries\', \'glasso\', \'qgraph\', \'reticulate\', \'keras\', \'rgl\', \'glmnet\'), repos=\'https://cran.rstudio.com\')"')
os.system("unzip Rsafd.zip")
os.system('R -e "install.packages(\'Rsafd\', repos = NULL, type=\'source\')"')

# hacky way tp get around flawed TclTk installs on MacOS
tcltk_path = Path(sys.exec_prefix,
                  "lib", "R", "library", "tcltk", "libs", "tcltk.so")
if not tcltk_path.exists():
    os.system("cp {} {}".format(tcltk_path.with_suffix(".dylib"), tcltk_path))
