#!/bin/usr/env python
import glob
import sys
import os

from setuptools import setup, find_namespace_packages

packages = find_namespace_packages(include=['pgscen.*'])

setup(name='pgscen',
      version='0.1',
      description='Power grid scenario creation platform for load and production of wind and solar',
      author='Xinshuo Yang',
      author_email='xy3134@princeton.edu',
      packages=packages,
      install_requires=['rpy2','numpy','matplotlib','pandas','scipy','jupyter','openpyxl']                    
     )