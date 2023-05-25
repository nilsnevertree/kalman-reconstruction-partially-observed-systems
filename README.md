kalman-reconstruction-partially-observered-systems
==============================

Data-driven Reconstruction of Partially Observed Dynamical Systems using Kalman algorithms and an itterative procedure.

## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

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

### Applying the Lorenz Model:
````python

result = pipeline.run_Kalman_SEM_to_xarray(
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
  * kalman_itteration  (kalman_itteration) int32 0 1 2 3 4 5 6 7 8 9
Data variables:
    states             (time, state_names) float64 0.1254 28.92 ... -6.976
    uncertainties      (time, state_names, state_names_copy) float64 0.00548 ...
    M                  (state_names, state_names_copy) float64 0.9994 ... 0.9288
    Q                  (state_names, state_names_copy) float64 0.01254 ... 3.3
    log_likelihod      (kalman_itteration) float64 -2.733e+03 ... -888.1

````

## Pre-commit
In order to use linting, pre-commit is used in this repository.
To lint your code, install [pre-commit](https://pre-commit.com/) and run ``pre-commit run --all-files`` before each commit.
This takes care of formating all files using the configuration from [.pre-commit-config.yaml](.pre-commit-config.yaml).

Please note that the https://github.com/kynan/nbstripout is used to make sure that the output from notebooks is cleared.
To disable it, comment out the part in [.pre-commit-config.yaml](.pre-commit-config.yaml?plain=1#L65)

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
