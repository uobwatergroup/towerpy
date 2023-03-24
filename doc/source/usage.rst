Usage
=====

.. _installation:

Installation
------------

To use towerpy, first install it using:

.. code-block:: console

   (.venv) $ git clone https://github.com/uobwatergroup/towerpy.git
   (.venv) $ python -m pip install -e .

Creating recipes
----------------

To initialise a radar object you can use the
``towerpy.io.ukmo.Rad_scan()`` class:

.. autofunction:: towerpy.io.ukmo.Rad_scan

The ``towerpy.io.ukmo.Rad_scan.ppi_ukmoraw()`` function retrieves raw
polarimetric variables from the current UKMO PPI binary files.

.. autofunction:: towerpy.io.ukmo.Rad_scan.ppi_ukmoraw

For example:

>>> import towerpy as tp
>>> tp.io.ukmo.Rad_scan


