"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

from setuptools import setup, find_packages
from os import path

setup(
      name='towerpy',
      version='0.99-alpha',
      description='Toolbox for processing polarimetric weather radar data',
      url='https://github.com/uobwatergroup/towerpy',
      author='Daniel Sanchez-Rivas and Miguel A Rico-Ramirez',
      author_email='dsanche1@uni-bonn.de; M.A.Rico-Ramirez@bristol.ac.uk',
      python_requires=">=3.9",
      packages=find_packages(exclude=['datasets']),
      include_package_data=True,
      install_requires=[
            "numpy",
            "matplotlib",
            # "cartopy>=0.19",
            # "matplotlib==3.4.2",
            # "scipy",
      ],
)
