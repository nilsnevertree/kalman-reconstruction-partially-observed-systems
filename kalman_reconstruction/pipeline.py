import numpy as np
import xarray as xr

from kalman_reconstruction.kalman import Kalman_SEM
from kalman_reconstruction.kalman_time_dependent import Kalman_SEM_time_dependent


def run_Kalman_SEM(y_list, random_list, nb_iter_SEM=30, variance_obs_comp=0.0001):
    """
    Run the Kalman Stochastic Expectation-Maximization (SEM) algorithm.

    Parameters:
    ----------
    y_list : list
        List of observed variables.
        Varibales need to be 1D numpy arrays.
    random_list : list
        List of random variables.
        Varibales need to be 1D numpy arrays.
    nb_iter_SEM : int
        Number of iterations for the SEM algorithm.
        Default is 30.
    variance_obs_comp :float
        Variance of observation components.
        Default is 0.0001.

    Returns:
    ----------
    tuple :
        A tuple containing the following elements:
        - x_s (ndarray): Array of estimated states with shape (T, n), where T is the number of time steps and n is the number of states.
        - P_s (ndarray): Array of estimated state uncertainties with shape (T, n, n).
        - M (ndarray): Array of estimated state transition matrix with shape (n, n).
        - log_likelihood (float): Log-likelihood of the estimated model.
        - x (ndarray): Array of estimated states for each iteration with shape (nb_iter_SEM, T, n).
        - x_f (ndarray): Array of forecasted states for each iteration with shape (nb_iter_SEM, T, n).
        - Q (ndarray): Array of estimated process noise covariance matrix with shape (n, n).

    Note:
    ----------
        - The input arrays y_list and random_list should have the same length and represent the observed variables and random variables, respectively.
    """
    # state
    y = np.array(y_list).T
    x_list = y_list + random_list
    x = np.array(x_list).T

    # shapes
    n = np.shape(x)[1]
    p = np.shape(y)[1]

    # kalman parameters
    H = np.append(np.eye(p), np.zeros((p, n)), axis=1)[:, 0:n]
    R = variance_obs_comp * np.eye(p)

    # stochastic EM
    return Kalman_SEM(x, y, H, R, nb_iter_SEM)


def run_Kalman_SEM_to_xarray(
    ds,
    state_variables,
    random_variables,
    nb_iter_SEM=30,
    variance_obs_comp=0.0001,
    suffix="",
):
    """
    Run the Kalman SEM algorithm on the input dataset and return the results in
    an xarray Dataset.

    This function applies the Kalman_SEM algorithm on the specified state and random variables in the given xarray dataset.
    It performs a specified number of iterations and computes the state estimates, uncertainties, transition matrix, and
    process noise covariance matrix. The results are stored in a new xarray dataset.

    Uses the run_Kalman_SEM() algorithm.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing the state and random variables.
    state_variables : list
        List of state variables to be used in the Kalman SEM algorithm.
    random_variables : list
        List of random variables to be used in the Kalman SEM algorithm.
    nb_iter_SEM : int, optional
        Number of iterations for the Kalman SEM algorithm.
        Default is 30.
    variance_obs_comp : float, optional
        Variance parameter for observation components in the Kalman SEM algorithm.
        Default is 0.0001.
    suffix : str, optional
        Suffix to be appended to the variable names in the output dataset.
        Default is an empty string.

    Returns
    -------
    xarray.Dataset
        New xarray dataset containing the results of the Kalman_SEM algorithm.
        Dimensions are:

    Output Dimensions:
    - time: The time dimension of the input dataset.
    - state_names: Dimension representing the state variables.
    - state_names_copy: Dimension representing a copy of the state variables.

    Output Coordinates:
    - state_names: Coordinates corresponding to the state variables.
    - state_names_copy: Coordinates corresponding to the copy of the state variables.
    - kalman_itteration: Coordinates representing the Kalman SEM iteration index.

    Output Variables:
    - states_<suffix>: DataArray containing the estimated states over time.
        - Dimensions: time, state_names
    - uncertainties_<suffix>: DataArray containing the uncertainties of the estimated states over time.
        - Dimensions: time, state_names, state_names_copy
    - M_<suffix>: DataArray containing the transition matrix M.
        - Dimensions: state_names, state_names_copy
    - Q_<suffix>: DataArray containing the observation noise covariance matrix Q.
        - Dimensions: state_names, state_names_copy
    - log_likelihod_<suffix>: DataArray containing the log-likelihood values for each Kalman SEM iteration.
        - Dimensions: kalman_itteration

    Note:
    - The output variable names will be suffixed with the provided suffix parameter.
    - If your given xr.DataSet was provided using a selection by values or indices, it is suggested to use the expand_dims_coords() function in order to contain the correct values of the dimensions and coordinates.
    """

    # function to craete new names from key and krn
    join_names = lambda l: "".join(l)

    observation_variables = list(state_variables) + list(random_variables)

    # create a list of numpy array to be used for the Kalman itteration
    y_list = []
    random_list = []
    for var in state_variables:
        y_list.append(ds[var].values.flatten())
    for var in random_variables:
        random_list.append(ds[var].values.flatten())

    # ---------------
    # run the Kalman_SEM algorithm
    # ---------------
    (
        x_s,
        P_s,
        M,
        log_likelihod,
        x,
        x_f,
        Q,
    ) = run_Kalman_SEM(
        y_list=y_list,
        random_list=random_list,
        nb_iter_SEM=nb_iter_SEM,
        variance_obs_comp=variance_obs_comp,
    )

    # ---------------
    # create result dataset in which to store the whole lists
    # ---------------
    result = xr.Dataset({})

    # store x_s
    new_var = join_names(["states", suffix])
    result[new_var] = xr.DataArray(
        data=x_s,
        dims=["time", "state_names"],
    )
    # store P_s
    new_var = join_names(["uncertainties", suffix])
    result[new_var] = xr.DataArray(
        data=P_s,
        dims=["time", "state_names", "state_names_copy"],
    )
    # store M and Q
    result = result.assign_coords(
        dict(
            state_names=observation_variables,
            state_names_copy=observation_variables,
        )
    )
    # store M
    new_var = join_names(["M", suffix])
    result[new_var] = xr.DataArray(
        data=M,
        dims=["state_names", "state_names_copy"],
    )
    # store Q
    new_var = join_names(["Q", suffix])
    result[new_var] = xr.DataArray(
        data=Q,
        dims=["state_names", "state_names_copy"],
    )
    # store the log_likelihod
    result = result.assign_coords(dict(kalman_itteration=np.arange(nb_iter_SEM)))
    new_var = join_names(["log_likelihod", suffix])
    result[new_var] = xr.DataArray(
        data=log_likelihod,
        dims=["kalman_itteration"],
    )

    return result


def run_Kalman_SEM_time_dependent(
    y_list, random_list, nb_iter_SEM=30, variance_obs_comp=0.0001
):
    """
    Run the Kalman Stochastic Expectation-Maximization (SEM) algorithm.

    Parameters:
    ----------
    y_list : list
        List of observed variables.
        Varibales need to be 1D numpy arrays.
    random_list : list
        List of random variables.
        Varibales need to be 1D numpy arrays.
    nb_iter_SEM : int
        Number of iterations for the SEM algorithm.
        Default is 30.
    variance_obs_comp :float
        Variance of observation components.
        Default is 0.0001.

    Returns:
    ----------
    tuple :
        A tuple containing the following elements:
        - x_s (ndarray): Array of estimated states with shape (T, n), where T is the number of time steps and n is the number of states.
        - P_s (ndarray): Array of estimated state uncertainties with shape (T, n, n).
        - M (ndarray): Array of estimated state transition matrix with shape (T, n, n).
        - log_likelihood (float): Log-likelihood of the estimated model.
        - x (ndarray): Array of estimated states for each iteration with shape (nb_iter_SEM, T, n).
        - x_f (ndarray): Array of forecasted states for each iteration with shape (nb_iter_SEM, T, n).
        - Q (ndarray): Array of estimated process noise covariance matrix with shape (T, n, n).

    Note:
    ----------
        - The input arrays y_list and random_list should have the same length and represent the observed variables and random variables, respectively.
    """
    # state
    y = np.array(y_list).T
    x_list = y_list + random_list
    x = np.array(x_list).T

    # shapes
    n = np.shape(x)[1]
    p = np.shape(y)[1]

    # kalman parameters
    H = np.append(np.eye(p), np.zeros((p, n)), axis=1)[:, 0:n]
    R = variance_obs_comp * np.eye(p)

    # stochastic EM
    return Kalman_SEM_time_dependent(x, y, H, R, nb_iter_SEM)


def xarray_Kalman_SEM_time_dependent(
    ds,
    state_variables,
    random_variables,
    nb_iter_SEM=30,
    variance_obs_comp=0.0001,
    suffix="",
):
    """
    Run the Kalman SEM algorithm on the input dataset and return the results in
    an xarray Dataset.

    This function applies the Kalman_SEM algorithm on the specified state and random variables in the given xarray dataset.
    It performs a specified number of iterations and computes the state estimates, uncertainties, transition matrix, and
    process noise covariance matrix. The results are stored in a new xarray dataset.

    Uses the run_Kalman_SEM() algorithm.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing the state and random variables.
    state_variables : list
        List of state variables to be used in the Kalman SEM algorithm.
    random_variables : list
        List of random variables to be used in the Kalman SEM algorithm.
    nb_iter_SEM : int, optional
        Number of iterations for the Kalman SEM algorithm.
        Default is 30.
    variance_obs_comp : float, optional
        Variance parameter for observation components in the Kalman SEM algorithm.
        Default is 0.0001.
    suffix : str, optional
        Suffix to be appended to the variable names in the output dataset.
        Default is an empty string.

    Returns
    -------
    xarray.Dataset
        New xarray dataset containing the results of the Kalman_SEM algorithm.
        Dimensions are:

    Output Dimensions:
    - time: The time dimension of the input dataset.
    - state_names: Dimension representing the state variables.
    - state_names_copy: Dimension representing a copy of the state variables.

    Output Coordinates:
    - state_names: Coordinates corresponding to the state variables.
    - state_names_copy: Coordinates corresponding to the copy of the state variables.
    - kalman_itteration: Coordinates representing the Kalman SEM iteration index.

    Output Variables:
    - states_<suffix>: DataArray containing the estimated states over time.
        - Dimensions: time, state_names
    - uncertainties_<suffix>: DataArray containing the uncertainties of the estimated states over time.
        - Dimensions: time, state_names, state_names_copy
    - M_<suffix>: DataArray containing the transition matrix M.
        - Dimensions: state_names, state_names_copy
    - Q_<suffix>: DataArray containing the observation noise covariance matrix Q.
        - Dimensions: state_names, state_names_copy
    - log_likelihod_<suffix>: DataArray containing the log-likelihood values for each Kalman SEM iteration.
        - Dimensions: kalman_itteration

    Note:
    - The output variable names will be suffixed with the provided suffix parameter.
    - If your given xr.DataSet was provided using a selection by values or indices, it is suggested to use the expand_dims_coords() function in order to contain the correct values of the dimensions and coordinates.
    """

    # function to craete new names from key and krn
    join_names = lambda l: "".join(l)

    observation_variables = list(state_variables) + list(random_variables)

    # create a list of numpy array to be used for the Kalman itteration
    y_list = []
    random_list = []
    for var in state_variables:
        y_list.append(ds[var].values.flatten())
    for var in random_variables:
        random_list.append(ds[var].values.flatten())

    # ---------------
    # run the Kalman_SEM algorithm
    # ---------------
    (
        x_s,
        P_s,
        M,
        log_likelihod,
        x,
        x_f,
        Q,
    ) = run_Kalman_SEM_time_dependent(
        y_list=y_list,
        random_list=random_list,
        nb_iter_SEM=nb_iter_SEM,
        variance_obs_comp=variance_obs_comp,
    )
    # ---------------
    # create result dataset in which to store the whole lists
    # ---------------
    result = xr.Dataset({})

    # store x_s
    new_var = join_names(["states", suffix])
    result[new_var] = xr.DataArray(
        data=x_s,
        dims=["time", "state_names"],
    )
    # store P_s
    new_var = join_names(["uncertainties", suffix])
    result[new_var] = xr.DataArray(
        data=P_s,
        dims=["time", "state_names", "state_names_copy"],
    )
    # store M and Q
    result = result.assign_coords(
        dict(
            state_names=observation_variables,
            state_names_copy=observation_variables,
        )
    )
    # store M
    new_var = join_names(["M", suffix])
    result[new_var] = xr.DataArray(
        data=M,
        dims=["time", "state_names", "state_names_copy"],
    )
    # store Q
    new_var = join_names(["Q", suffix])
    result[new_var] = xr.DataArray(
        data=Q,
        dims=["time", "state_names", "state_names_copy"],
    )
    # store the log_likelihod
    result = result.assign_coords(dict(kalman_itteration=np.arange(nb_iter_SEM)))
    new_var = join_names(["log_likelihod", suffix])
    result[new_var] = xr.DataArray(
        data=log_likelihod,
        dims=["kalman_itteration"],
    )

    return result


def add_random_variable(ds, var_name, random_generator, variance, dim="time"):
    """
    Add a random variable to a given xarray dataset.

    Parameters:
    -----------
    ds : xarray.Dataset
        The input dataset to which the random variable will be added.
    var_name : str
        The name of the random variable.
    random_generator : numpy.random.Generator
        The random number generator to generate the random variable values.
    variance : float
        The variance of the random variable.
    dim : str, optional
        The dimension along which to add the random variable. Default is "time".

    Returns:
    --------
    None

    Notes:
    ------
    - The random variable values are generated using the provided `random_generator`
      by drawing samples from a normal distribution with mean 0 and the specified `variance`.
    - The random variable is added as a new DataArray to the input dataset `ds`,
      with the name `var_name` and the specified dimension `dim`.
    - The coordinates of the added DataArray are set to the coordinate values of the dimension `dim` in `ds`.
    """

    ds[var_name] = xr.DataArray(
        data=random_generator.normal(loc=0, scale=variance, size=len(ds[dim])),
        dims=[dim],
    )


def expand_dims_coords(
    ds1: xr.Dataset, ds2: xr.Dataset, select_dict: dict
) -> xr.Dataset:
    """
    Expand dimensions of ds1 and assign coordinates from ds2.

    This function expands the dimensions of ds1 using the expand_dims method and assigns the coordinates
    from ds2 to the expanded dataset. The resulting dataset contains the expanded dimensions from ds1
    along with the coordinates from ds2.

    Parameters:
        ds1 (xr.Dataset): The dataset to expand dimensions.
        ds2 (xr.Dataset): The dataset containing coordinates to assign.
        select_dict (dict): Dictonary of selection or index selection containing names and values of the coords and dims to expand.

    Returns:
        xr.Dataset: The dataset with expanded dimensions and assigned coordinates.

    Example:
    >>> # Create example datasets
    >>> ds1 = xr.Dataset(
        {'var1': (('x', 'y'), np.random.rand(10, 20))},
        coords={'x': range(10), 'y': range(20)}
    )
    >>> ds1

    <xarray.Dataset>
    Dimensions:  (x: 10, y: 20)
    Coordinates:
    * x        (x) int32 0 1 2 3 4 5 6 7 8 9
    * y        (y) int32 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
    Data variables:
    var1     (x, y) float64

    >>> ds2 = xr.Dataset(
        {'var2': ('z', np.random.rand(5))},
        coords={'z': range(5)}
    )
    >>> ds2
    <xarray.Dataset>
    Dimensions:  (z: 5)
    Coordinates:
    * z        (z) int32 0 1 2 3 4
    Data variables:
        var2     (z) float64

    >>> select_dict = {'z': 2}
    >>> # Call the function
    >>> result = expand_dims_coords(ds1, ds2, select_dict)
    >>> print(result)
    <xarray.Dataset>
    Dimensions:  (z: 1, x: 10, y: 20)
    Coordinates:
    * x        (x) int32 0 1 2 3 4 5 6 7 8 9
    * y        (y) int32 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
    * z        (z) int32 2
    Data variables:
        var1     (z, x, y) float64
    """
    selection_vars = list(select_dict.keys())
    ds1 = ds1.drop_vars(selection_vars, errors="ignore")
    try:
        ds1_expanded = ds1.assign_coords(**ds2.sel(select_dict).coords)
    except Exception as E:
        try:
            ds1_expanded = ds1.assign_coords(**ds2.isel(select_dict).coords)
        finally:
            pass
    ds1_expanded = ds1_expanded.expand_dims(selection_vars)

    return ds1_expanded