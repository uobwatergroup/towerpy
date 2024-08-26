"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

from setuptools import setup, find_packages
from os import path

setup(
      name='towerpy-rd',
      version='1.0.5',
      description='Toolbox for processing polarimetric weather radar data',
      url='https://github.com/uobwatergroup/towerpy',
      author='Daniel Sanchez-Rivas and Miguel A Rico-Ramirez',
      author_email='towerpy@icloud.com, M.A.Rico-Ramirez@bristol.ac.uk',
      python_requires=">=3.9",
      packages=find_packages(exclude=['datasets', 'docs']),
      include_package_data=True,
      package_data={'towerpy': ['io/lnxlibreadpolarradardata.so', 'eclass/lnxlibclutterclassifier.so', 'attc/lnxlibattenuationcorrection.so',
      				'io/w64libreadpolarradardata.dll', 'eclass/w64libclutterclassifier.dll', 'attc/w64libattenuationcorrection.dll']},
      install_requires=['numpy', 'matplotlib>=3.5.2', 'scipy', 'cartopy', 'netCDF4'],
      classifiers=(
        "Programming Language :: Python :: 3",
	"Programming Language :: Python :: 3.9",
    	"Programming Language :: Python :: 3.10",
    	"Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",),
      )
