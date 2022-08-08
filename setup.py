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
                             '../data/MetaData/*.xlsx',
                             '../data/MetaData/*.zip']},

    entry_points={
        'console_scripts': [
            'pgscen-load=pgscen.command_line:create_load_scenarios',
            'pgscen-wind=pgscen.command_line:create_wind_scenarios',
            'pgscen-solar=pgscen.command_line:create_solar_scenarios',
            'pgscen-load-solar'
            '=pgscen.command_line:create_load_solar_scenarios',
            'pgscen=pgscen.command_line:create_scenarios',

            'pgscen-pca-solar=pgscen.command_line:create_pca_solar_scenarios',
            'pgscen-pca-load-solar'
            '=pgscen.command_line:create_load_solar_scenarios',
            'pgscen-pca=pgscen.command_line:create_pca_scenarios',

            'pgscen-rts=pgscen.rts_gmlc.command_line:create_scenarios',
            'pgscen-rts-joint'
            '=pgscen.rts_gmlc.command_line:create_joint_scenarios',
            'pgscen-rts-pca-solar'
            '=pgscen.rts_gmlc.command_line:create_pca_solar_scenarios',
            'pgscen-rts-pca-load-solar'
            '=pgscen.rts_gmlc.command_line:create_pca_load_solar_scenarios',
            'pgscen-rts-pca=pgscen.rts_gmlc.command_line:create_pca_scenarios',
            ],
        },

    install_requires=['numpy', 'matplotlib', 'pandas', 'scipy'],
    )


os.system("curl https://carmona.princeton.edu/SVbook/Rsafd.zip "
          "--output Rsafd.zip")
os.system("unzip Rsafd.zip")
os.system('R -e "install.packages(\'Rsafd\', repos = NULL, type=\'source\')"')
os.system("rm -rf Rsafd Rsafd.zip")

# hacky way tp get around flawed TclTk installs on MacOS
tcltk_path = Path(sys.exec_prefix,
                  "lib", "R", "library", "tcltk", "libs", "tcltk.so")
if not tcltk_path.exists():
    os.system("cp {} {}".format(tcltk_path.with_suffix(".dylib"), tcltk_path))
