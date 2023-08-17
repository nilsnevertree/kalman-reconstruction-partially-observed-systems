kalman-reconstruction-partially-observered-systems
==============================
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Documentation Status](https://readthedocs.org/projects/kalman-reconstruction-partially-observed-systems/badge/?version=latest)](https://kalman-reconstruction-partially-observed-systems.readthedocs.io/en/latest/?badge=latest)

Data-driven Reconstruction of Partially Observed Dynamical Systems using Kalman algorithms and an iterative procedure.

## Project Organization
------------
    ├───.github
    │   └───workflows                           <- GitHub workflows
    ├───ci                                      <- Environments for the GitHub workflows
    ├───data                                    <- Data should go here
    ├───docs                                    <- Documentation
    ├───kalman_reconstruction                   <- The main library that is build
    │   │   cli.py
    │   │   custom_plot.py                      <- Customized plot functions to visualize data
    │   │   example_models.py                   <- Example numerical models to create data. E.g. Lorenz63 model
    │   │   kalman.py                           <- Module of Kalman algorithms
    │   │   kalman_time_dependent.py            <- Module of Kalman algorithms taking timedependency into account (Local Linear Regression)
    │   │   pipeline.py                         <- Pipeline which encapsuals the Kalman module to be used with xarray see also an example below
    │   │   reconstruction.py                   <- Module to compute a reconstruction of the Lorenz63 hidden variables from latent variabels
    │   │   statistics.py                       <- Module containing statistical algorithms
    │   │   _version.py
    │   │   __init__.py
    │   │   __main__.py
    ├───notebooks                               <- Store notebooks here. A typical format for user ``Adam Bashfort`` on the topic ``stability`` would be 01_AB_stability.ipynb
    ├───temporary                               <- Folder to store files which shall not be traced.
    └───tests
--------
## Installation
- Install directly from GitHub using pip:
``pip install git+https://github.com/nilsnevertree/kalman-reconstruction-partially-observered-systems``
- Clone the repository and run ``python -m pip install . -e`` in the repository folder.

## Pipeline Usage
The ``pipeline`` module allows a easy application of the Kalman algorithms on [xarray.DataSet](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) datatypes. These were chosen to handle multidimensional data created during the calculations.
The following example should show how to apply the Kalman SEM algorithm to an example simulation of the Lorenz63 model.

##### Imports
````python
import numpy as np
from kalman_reconstruction import pipeline
from kalman_reconstruction import example_models
````
##### Creating data using the Lorenz63 model
````python
>>> data = example_models.Lorenz_63_xarray(dt=0.01, time_length=5, time_steps=None)
>>> seed = 345831200837
>>> variance = 5
>>> rng1 = np.random.default_rng(seed=seed)
>>> rng2 = np.random.default_rng(seed=seed + 1)
>>> rng3 = np.random.default_rng(seed=seed + 2)
>>> rng4 = np.random.default_rng(seed=seed + 3)
>>> pipeline.add_random_variable(
...     data, var_name="z1", random_generator=rng1, variance=variance
... )
>>> pipeline.add_random_variable(
...      data, var_name="z2", random_generator=rng2, variance=variance
... )
>>> pipeline.add_random_variable(
...     data, var_name="z3", random_generator=rng3, variance=variance
... )
````
##### Data:
````python
print(data)
<xarray.Dataset>
Dimensions:  (time: 500)
Coordinates:
  * time     (time) float64 0.0 0.01 0.02 0.03 0.04 ... 4.95 4.96 4.97 4.98 4.99
Data variables:
    x1       (time) float64 8.0 7.232 6.53 5.892 ... 4.36 4.688 5.043 5.425
    x2       (time) float64 0.0 -0.1217 -0.177 -0.1789 ... 8.098 8.727 9.399
    x3       (time) float64 30.0 29.21 28.43 27.67 ... 13.32 13.32 13.37 13.49
    z1       (time) float64 -6.395 3.449 10.29 1.823 ... -4.295 6.481 1.779 9.81
    z2       (time) float64 -1.226 9.979 -9.211 3.195 ... 3.434 2.342 -7.697
    z3       (time) float64 -1.882 -1.183 -1.002 -3.343 ... -2.358 -2.183 -9.06
Attributes:
    description:  Lorenz Model.
````

##### Applying the Kalman_SEM wrapper ``xarray_Kalman_SEM``:
````python
>>> result = pipeline.xarray_Kalman_SEM(
...     ds=data,
...     observation_variables=["x2", "x3"],
...     state_variables=["x2", "x3", "z1", "z2", "z3"],
...     nb_iter_SEM=10,
...     variance_obs_comp=0.0001,
... )
100%|███████| 10/10 [00:02<00:00,  4.71it/s]
````

##### Result:
````python
print(result)
<xarray.Dataset>
Dimensions:           (time: 500, state_name: 5, state_name_copy: 5,
                       kalman_iteration: 10)
Coordinates:
  * time              (time) float64 0.0 0.01 0.02 0.03 ... 4.96 4.97 4.98 4.99
  * state_name        (state_name) <U2 'x2' 'x3' 'z1' 'z2' 'z3'
  * state_name_copy   (state_name_copy) <U2 'x2' 'x3' 'z1' 'z2' 'z3'
  * kalman_iteration  (kalman_iteration) int32 0 1 2 3 4 5 6 7 8 9
Data variables:
    states            (time, state_name) float64 0.1279 28.92 ... -1.559 -6.978
    covariance        (time, state_name, state_name_copy) float64 0.005467 .....
    M                 (state_name, state_name_copy) float64 0.9994 ... 0.9291
    Q                 (state_name, state_name_copy) float64 0.0125 ... 3.291
    log_likelihod     (kalman_iteration) float64 -2.734e+03 -2.52e+03 ... -888.2
````
The ``states`` variable contains the estimate of the Kalman_SEM and the ``covariance`` the corresponding covariance matrices for each timestep.
A global estiamtion of the model Matrix ``M`` and Model Uncertainty ``Q`` are given too.
If the estimation for all iterations shall be returned, use the ``Kalman_SEM_full_output`` function.
The log-likelihood as a a function of the iterations is always stored.

##### Transform to readable Dataset format
To transform this into a Dataset similar to the input Dataset, one can use the ``pipeline.from_standard_dataset`` function.
Two examples are shown below:
````python
# States
states = pipeline.from_standard_dataset(result, var_name = "states", dim_name = "state_name")
print(states)

<xarray.Dataset>
Dimensions:           (time: 500, state_name_copy: 5, kalman_iteration: 10)
Coordinates:
  * time              (time) float64 0.0 0.01 0.02 0.03 ... 4.96 4.97 4.98 4.99
  * state_name_copy   (state_name_copy) <U2 'x2' 'x3' 'z1' 'z2' 'z3'
  * kalman_iteration  (kalman_iteration) int32 0 1 2 3 4 5 6 7 8 9
Data variables:
    x2                (time) float64 0.1279 -0.1225 -0.1761 ... 8.728 9.398
    x3                (time) float64 28.92 29.2 28.43 ... 13.32 13.37 13.49
    z1                (time) float64 0.2364 -0.1215 -2.109 ... 1.035 1.27 1.324
    z2                (time) float64 1.293 -0.7298 -4.571 ... -1.866 -1.559
    z3                (time) float64 0.5584 1.613 2.029 ... -6.272 -6.921 -6.978

# Covariances
covariances = pipeline.from_standard_dataset(result, var_name = "covariance", dim_name = "state_name")
print(covariances)

<xarray.Dataset>
Dimensions:           (time: 500, state_name_copy: 5, kalman_iteration: 10)
Coordinates:
  * time              (time) float64 0.0 0.01 0.02 0.03 ... 4.96 4.97 4.98 4.99
  * state_name_copy   (state_name_copy) <U2 'x2' 'x3' 'z1' 'z2' 'z3'
  * kalman_iteration  (kalman_iteration) int32 0 1 2 3 4 5 6 7 8 9
Data variables:
    x2                (time, state_name_copy) float64 0.005467 ... -0.0009999
    x3                (time, state_name_copy) float64 -0.004748 ... -0.0003233
    z1                (time, state_name_copy) float64 -0.001114 ... 5.442
    z2                (time, state_name_copy) float64 0.02026 ... -1.831
    z3                (time, state_name_copy) float64 0.03091 0.01474 ... 2.281

````

## Contribution
### Pre-commit
In order to use linting, pre-commit is used in this repository.
To lint your code, install [pre-commit](https://pre-commit.com/) and run ``pre-commit run --all-files`` before each commit.
This takes care of formatting all files using the configuration from [.pre-commit-config.yaml](.pre-commit-config.yaml).

Please note that the https://github.com/kynan/nbstripout is used to make sure that the output from notebooks is cleared.
To disable it, comment out the part in [.pre-commit-config.yaml](.pre-commit-config.yaml?plain=1#L65)

## Legal notes

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

The kalman algorithms in [kalman.py](./kalman_reconstruction/kalman.py) were provided by (Pierre Tandeo)[https://github.com/ptandeo] from the [Kalman](https://github.com/ptandeo/kalman) GitHub Repository, which was as of 2023-07-18 under GPL v3.

For questions, raise an issue or contact [nilsnevertree](https://github.com/nilsnevertree)

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
