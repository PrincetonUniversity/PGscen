#!/bin/usr/env python
import os
import sys
import setuptools
from pathlib import Path


setuptools.setup(
    name='pgscen',
    version='0.2.0-a1',
    description="power grid scenario creation platform for "
                "load and production of wind and solar",
    author='Xinshuo Yang',
    author_email='xy3134@princeton.edu',

    packages=['pgscen', 'pgscen.utils', 'pgscen.rts_gmlc'],
    package_data={'pgscen': ['../test/resources/*.p.gz']},

    entry_points={
        'console_scripts': [
            'pgscen-load=pgscen.command_line:run_load',
            'pgscen-wind=pgscen.command_line:run_wind',
            'pgscen-solar=pgscen.command_line:run_solar',
            'pgscen-load-solar=pgscen.command_line:run_load_solar_joint',

            'pgscen-rts=pgscen.rts_gmlc.command_line:run_rts',
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
