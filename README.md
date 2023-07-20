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
A more readable application and easy to use a ``pipeline`` is implemented.
For this purpose the [xarray.DataSet](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) datatype was chosen to handle multidimensional data created during the calculations.

The following example should show how to this.

### Imports
````python
from kalman_reconstruction import pipeline
from kalman_reconstruction import example_models
````
### Creating data using the Lorenz63 model
````python
data = example_models.Lorenz_63_xarray(dt=0.01, time_length=5, time_steps=None)
seed = 345831200837
variance = 5
rng1 = np.random.default_rng(seed=seed)
rng2 = np.random.default_rng(seed=seed + 1)
rng3 = np.random.default_rng(seed=seed + 2)
rng4 = np.random.default_rng(seed=seed + 3)
pipeline.add_random_variable(
    data, var_name="z1", random_generator=rng1, variance=variance
)
pipeline.add_random_variable(
    data, var_name="z2", random_generator=rng2, variance=variance
)
pipeline.add_random_variable(
    data, var_name="z3", random_generator=rng3, variance=variance
)
print(data)
````

### Data:
````python
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

### Applying the Kalman_SEM wrapper ``xarray_Kalman_SEM``:
````python

result = pipeline.xarray_Kalman_SEM(
    ds=data,
    state_variables=["x2", "x3"],
    random_variables=["z1", "z2", "z3"],
    nb_iter_SEM=10,
    variance_obs_comp=0.0001,
)
print(result)
````

### Result:
````python

<xarray.Dataset>
Dimensions:            (time: 500, state_names: 5, state_names_copy: 5,
                        kalman_itteration: 10)
Coordinates:
  * time               (time) float64 0.0 0.01 0.02 0.03 ... 4.96 4.97 4.98 4.99
  * state_names        (state_names) <U2 'x2' 'x3' 'z1' 'z2' 'z3'
  * state_names_copy   (state_names_copy) <U2 'x2' 'x3' 'z1' 'z2' 'z3'
  * kalman_iteration   (kalman_iteration) int32 0 1 2 3 4 5 6 7 8 9
Data variables:
    states             (time, state_names) float64 0.1254 28.92 ... -6.976
    uncertainties      (time, state_names, state_names_copy) float64 0.00548 ...
    M                  (state_names, state_names_copy) float64 0.9994 ... 0.9288
    Q                  (state_names, state_names_copy) float64 0.01254 ... 3.3
    log_likelihod      (kalman_itteration) float64 -2.733e+03 ... -888.1

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

The kalman algorithms in [kalman.py](./kalman_reconstruction/kalman.py) were provided by [Pierre Tandeo](https://github.com/ptandeo) from the [Kalman](https://github.com/ptandeo/kalman) GitHub Repository, which was as of 2023-07-18 under GPL v3.

For questions, raise an issue or contact [nilsnevertree](https://github.com/nilsnevertree)

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
