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

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: towerpy.io.ukmo.Rad_scan

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`towerpy.io.ukmo.Rad_scan`
will raise an exception.

.. autoexception:: towerpy.InvalidKindError

For example:

>>> import towerpy as tp
>>> tp.io.ukmo.Rad_scan


