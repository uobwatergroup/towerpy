<p align="center">
  <img src="https://github.com/uobwatergroup/towerpy/blob/main/towerpy/towerpy_logosd.png?raw=true" alt="[towerpylogo]"/>
</p>

# :satellite: TOWERPY
Towerpy is an open-source toolbox designed for reading, processing and displaying polarimetric weather radar data.

<p align="left">
  <img alt="Platforms" src="https://img.shields.io/badge/ =&nbsp&nbsp OS &nbsp&nbsp&nbsp;=-critical?style=for-the-badge" />
  <img alt="linux" src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black"/>
  <img alt="wdw" src="https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white"/>
</p>
<p align="left">
  <img alt="tools" src="https://img.shields.io/badge/= Tools =-critical?style=for-the-badge" />
  <img alt="plang1" src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img alt="plang2" src="https://img.shields.io/badge/C-00599C?style=for-the-badge&logo=c&logoColor=white"/>
  <img alt="pt1" src="https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white"/> 
  <img alt="pt5" src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img alt="pt6" src="https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white"/>
</p>
<p align="left">
  <img alt="info" src="https://img.shields.io/badge/=&nbsp Info &nbsp;=-critical?style=for-the-badge"/>
  <a href="https://github.com/uobwatergroup/towerpy/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-GPL%20v3.0-yellow.svg?style=for-the-badge&logo=gnuprivacyguard"/></a>
  <img alt="GitHub release (latest SemVer)" src="https://img.shields.io/github/v/release/uobwatergroup/towerpy?style=for-the-badge">
  <a href="https://doi.org/10.1016/j.envsoft.2023.105746"><img src="https://img.shields.io/badge/DOI-10.1016/j.envsoft.2023.105746-important?style=for-the-badge&logo=creativecommons"/></a>
  
</p>


# :books: Documentation
[![Documentation Status](https://readthedocs.org/projects/towerpy/badge/?version=latest)](https://towerpy.readthedocs.io/en/latest/?badge=latest)

You can find more details about _Towerpy_ [here](https://towerpy.readthedocs.io/en/latest/).

# :speech_balloon: Citing
If you find _towerpy_ useful for a scientific publication, please consider citing the [towerpy paper](https://doi.org/10.1016/j.envsoft.2023.105746):

BibTeX:

       @article{sanchezrivas2023, title = {{Towerpy: An open-source toolbox for processing polarimetric weather radar data}},
       journal = {Environmental Modelling & Software}, pages= {105746}, year = {2023}, issn = {1364-8152},
       doi = {https://doi.org/10.1016/j.envsoft.2023.105746}, author = {Daniel Sanchez-Rivas and Miguel Angel Rico-Ramirez},
       keywords = {Weather radar, Polarimetry, Radar QPE, Radar research applications, Open source}}

# :hammer: Installing towerpy
To ensure a smooth installation of towerpy, we strongly recommend using **conda**, the open-source Python package management system. [Download](https://www.anaconda.com/) and install Anaconda to get conda and other essential data science and machine learning packages.

Once Anaconda is installed and working, we recommend adding the [conda-forge channel](https://conda-forge.org/) to the top of the channel list::

    conda config --add channels conda-forge

Strict channel priority can dramatically speed up conda operations and also reduce package incompatibility problems. We recommend setting channel priority to "strict" using::

    conda config --set channel_priority strict

Create a new environment to install towerpy::
    
    conda create -n towerpy python=3.10

Activate the new environment::

    conda activate towerpy

Install the latest version of Towerpy by choosing ONE of the following options:

(1) Downloading and installing Towerpy from Anaconda.org::
    
    conda install -c towerpy towerpy

(2) Downloading and installing Towerpy from GitHub (here we have the latest release)::    

    python -m pip install --upgrade git+https://github.com/uobwatergroup/towerpy.git@main

(3) Downloading and installing Towerpy from the Python Package Index (PyPI)::
    
    python -m pip install towerpy-rd

Finally, install other additional dependencies helpful for scientific computing::

    conda install -c conda-forge ipython jupyter sympy spyder scikit-learn wradlib


## Installing from source

Optionally, you can get the towerpy source code from the [GitHub repository](https://github.com/uobwatergroup/towerpy). 

Either download and unpack the zip file of the source code or use git to clone the repository::

    git clone https://github.com/uobwatergroup/towerpy.git

Once inside the folder where _towerpy_ was downloaded, it can be installed by using pip::

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


# :thought_balloon: Missing a specific feature? Found a bug?
Feel free to post a message to our [discussions page](https://github.com/uobwatergroup/towerpy/discussions) or have a look at our [contribution guidelines](.github/CONTRIBUTING.md) to find out about our coding standards.

# :construction_worker: Team
Towerpy is created and maintained by 
[Daniel Sanchez-Rivas](https://scholar.google.com/citations?user=NQSB5-8AAAAJ&hl=en)<a itemprop="sameAs" content="https://orcid.org/0000-0001-9356-6641" href="https://orcid.org/0000-0001-9356-6641" target="orcid.widget" rel="me noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon"></a></div> and [Miguel A Rico-Ramirez](https://research-information.bris.ac.uk/en/persons/miguel-a-rico-ramirez)<a itemprop="sameAs" content="https://orcid.org/0000-0002-8885-4582" href="https://orcid.org/0000-0002-8885-4582" target="orcid.widget" rel="me noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon"></a></div>

<p align="center">
  <img alt="GitHub User's stars" src="https://img.shields.io/github/stars/uobwatergroup?style=social"><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/uobwatergroup/towerpy">
</p>

[![Sparkline](https://stars.medv.io/uobwatergroup/towerpy.svg)](https://stars.medv.io/uobwatergroup/towerpy)

