.. kalman-reconstruction-partially-observered-systems documentation master file, created by
   sphinx-quickstart.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Modules and Code
=======================================================
Here you can find the documentation of the code used in this repository.

Main Modules:
-------------

The pipeline module contains the main functions to run the reconstruction using xarray.
The Kalman and Kalman Time Dependent modlues contain the Kalman algorithms on numpy base.

.. toctree::
   :maxdepth: 1
   :caption: Modules:

   kalman
   kalman_time_dependent
   pipeline

Further Modules:
-------------

The further modules can be used to generate example data and analyse and visualize the results.

.. toctree::
    :maxdepth: 1
    :caption: Modules:
    
    statistics
    example_models
    reconstruction
    custom_plot

Indices and tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`