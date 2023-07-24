.. _installation:

Installation
============

To ensure a smooth installation of towerpy, we strongly recommend using conda, the open-source Python package management system. `Download <https://www.anaconda.com/>`_ and install Anaconda to get conda and other essential data science and machine learning packages.

Once Anaconda is installed and working, we recommend adding the `conda-forge channel <https://conda-forge.org/>`_ to the top of the channel list::

    conda config --add channels conda-forge

Strict channel priority can dramatically speed up conda operations and also reduce package incompatibility problems. We recommend setting channel priority to "strict" using::

    conda config --set channel_priority strict

Create a new environment to install towerpy::

    conda create -n towerpy python=3.10

Activate the new environment::

    conda activate towerpy

Install the latest version of Towerpy by choosing ONE of the following options:

#. Downloading and installing Towerpy from Anaconda.org::

    conda install -c towerpy towerpy

#. Downloading and installing Towerpy from GitHub (here we have the latest release)::

    python -m pip install --upgrade git+https://github.com/uobwatergroup/towerpy.git@main

#. Downloading and installing Towerpy from the Python Package Index (PyPI)::
    
    python -m pip install towerpy-rd

Finally, install other additional dependencies helpful for scientific computing::

    conda install -c conda-forge ipython jupyter sympy spyder scikit-learn wradlib tqdm


Installing from source
----------------------

Optionally, you can get the towerpy source code from the `GitHub repository <https://github.com/uobwatergroup/towerpy>`_.

Either download and unpack the zip file of the source code or use git to clone the repository::

    git clone https://github.com/uobwatergroup/towerpy.git

Once inside the folder where _Towerpy_ was downloaded, it can be installed by using pip::

    python -m pip install .

Or install Towerpy in “editable” mode::

    python -m pip install -e .


Dependencies
------------

The following dependencies are required to work with _towerpy_:

.. list-table:: Dependencies
   :widths: 20 20
   :header-rows: 1

   * - Package
     - Required
   * - NumPy
     - v1.21+
   * - SciPy
     - v1.7.1+
   * - Matplotlib
     - v3.5.2+
   * - Cartopy
     - v0.19+
   * - netCDF4
     - vv1.5.8+

Some modules within *towerpy* run using shared objects/dynamic link libraries (.so, .dll). These libraries are automatically installed during the *towerpy* installation.
