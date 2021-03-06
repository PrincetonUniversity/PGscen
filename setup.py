#!/bin/usr/env python
import os
import sys
import setuptools
from pathlib import Path


setuptools.setup(
    name='pgscen',
    version='0.2.1-a0',
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
            'pgscen-load=pgscen.command_line:run_load',
            'pgscen-wind=pgscen.command_line:run_wind',
            'pgscen-solar=pgscen.command_line:run_solar',
            'pgscen-load-solar=pgscen.command_line:run_load_solar_joint',
            'pgscen=pgscen.command_line:run_t7k',

            'pgscen-pca-solar=pgscen.pca_command_line:run_solar',
            'pgscen-pca-load-solar=pgscen.pca_command_line:run_load_solar',
            'pgscen-pca=pgscen.pca_command_line:run_t7k_pca',

            'pgscen-rts=pgscen.rts_gmlc.command_line:run_rts',
            'pgscen-rts-joint=pgscen.rts_gmlc.command_line:run_rts_joint',
            'pgscen-rts-pca-solar=pgscen.rts_gmlc.pca_command_line:run_solar',
            'pgscen-rts-pca-load-solar'
            '=pgscen.rts_gmlc.pca_command_line:run_load_solar',
            'pgscen-rts-pca=pgscen.rts_gmlc.pca_command_line:run_rts_pca',
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
