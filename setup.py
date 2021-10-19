#!/bin/usr/env python
import os
import setuptools


setuptools.setup(
    name='pgscen',
    version='0.2.0-a0',
    description="power grid scenario creation platform for "
                "load and production of wind and solar",
    author='Xinshuo Yang',
    author_email='xy3134@princeton.edu',
    packages=['pgscen', 'pgscen.utils'],

    entry_points = {
        'console_scripts': [
            'pgscen-load=pgscen.command_line:run_load',
            'pgscen-wind=pgscen.command_line:run_wind',
            'pgscen-solar=pgscen.command_line:run_solar',
            'pgscen-load-solar=pgscen.command_line:run_load_solar_joint',
            ],
        },

    install_requires=['rpy2', 'numpy', 'matplotlib', 'pandas',
                      'scipy', 'jupyter', 'openpyxl'],
    )


os.system("curl https://carmona.princeton.edu/SVbook/Rsafd.zip "
          "--output Rsafd.zip")
os.system("unzip Rsafd.zip")
os.system('R -e "install.packages(\'Rsafd\', repos = NULL, type=\'source\')"')
os.system('rm -rf Rsafd Rsafd.zip')

