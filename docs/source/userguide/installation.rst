.. _installation:

Installation
------------

To ensure a smooth installation of towerpy, we strongly recommend using conda, the open-source Python package management system. `Download <https://www.anaconda.com/>`_ and install Anaconda to get conda and other essential data science and machine learning packages.

Once Anaconda is installed and working, create a new environment to install towerpy::

    conda create -n towerpy python=3.10

Activate the new environment::

    conda activate towerpy

Then install the required dependencies and additional dependencies helpful for scientific computing::

    conda install -c conda-forge numpy matplotlib scipy ipython jupyter pandas netcdf4 sympy nose spyder cartopy metpy scikit-learn

Finally, install the latest version of towerpy using::

    python -m pip install --upgrade git+https://github.com/uobwatergroup/towerpy.git@main

Installing from source
----------------------

Optionally, you can get the towerpy source code from the `GitHub repository <https://github.com/uobwatergroup/towerpy>`_.

Either download and unpack the zip file of the source code or use git to clone the repository::

    git clone https://github.com/uobwatergroup/towerpy.git

Once inside the folder where _Towerpy_ was downloaded, it can be installed by using pip::

    python -m pip install .

Or install Towerpy in “editable” mode::

    python -m pip install -e .


# :snake: Dependencies

## The following dependencies are required to work with _towerpy_:

Package | Required
------------ | -------------
NumPy | v1.21+
SciPy | v1.7.1+
Matplotlib | v3.5.2+
Cartopy | v0.19+
netCDF4 | vv1.5.8+

\*Some modules within _towerpy_ run using shared objects/dynamic link libraries (.so, .dll). These libraries are automatically installed during the _towerpy_ installation.
