from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import xarray as xr

from kalman_reconstruction.kalman import Kalman_filter, Kalman_SEM, Kalman_smoother
from kalman_reconstruction.kalman_time_dependent import Kalman_SEM_time_dependent
from kalman_reconstruction.statistics import assert_ordered_subset, ordered_like


# Kalman_Functions


def input_arrays_combined(
    observation_list: Iterable[np.ndarray], random_list: Iterable[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input arrays of observations (y) and states (x) for the
    Kalman_algorithms. The state is a concatenation of the observation
    variables followed by the random variabels.

    Note:
    - All arrays in the list must be 1D and of equal length T.

    Parameters:
        observation_list (list) len(p): List of observation variables.
        random_list (list) len(r): List of latent variables that shall be appended.

    Returns:
        Tuple : A Tuple containing the input arrays for the Kalman_algorithms.
            - states : np.ndarray shape(T, p+r),
            - observations : np.ndarray shape(T, p)
    """
    states, observations = input_arrays(
        state_list=observation_list + random_list, observation_list=observation_list
    )
    return states, observations


def input_arrays(
    state_list: Iterable[np.ndarray], observation_list: Iterable[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input arrays of observations (y) and states (x) for the
    Kalman_algorithms.

    Note:
    - All arrays in the list must be 1D and of equal length T.

    Parameters:
        state_list (list) len(n): List of state variables.
        observation_list (list) len(p): List of observed variables.

    Returns:
        Tuple : A Tuple containing the input arrays for the Kalman_algorithms.
            - states : np.ndarray shape(T, n),
            - observations : np.ndarray shape(T, p)
    """
    observations = np.array(observation_list).T
    states = np.array(
        state_list,
    ).T
    return states, observations


def input_matrices_H_R(
    states: np.ndarray,
    observations: np.ndarray,
    variance_obs_comp: float = 0.0001,
    axis: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create default input observation matrices H and R for a Kalman algorithms.

    H will only contain values of 1 at corresponding positions

    Parameters:
        states (ndarray) shape(T,n): Array of state variables.
        observations (ndarray) shape(T,p): Array of observation variables.
        variance_obs_comp (float): Variance of observation component. Default to 0.0001.
        axis (float): axis along which the number of variables is defined. Default to 1.

    Returns:
        Tuple: A Tuple containing the input matrices (H, R).
        - H : np.ndarray, shape (p, n)
            First estimation of the observation matrix that maps the state space to the observation space.
        - R : np.ndarray, shape (p, p)
            First estimation of the observation noise covariance matrix.
    """
    # shapes
    n = np.shape(states)[axis]
    p = np.shape(observations)[axis]

    # kalman parameters
    H = np.append(np.eye(p), np.zeros((p, n)), axis=axis)[:, 0:n]
    R = variance_obs_comp * np.eye(p)

    return H, R


def _input_matrices_H_R_from_n_p(
    n: int,
    p: int,
    variance_obs_comp: float = 0.0001,
    axis: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    # kalman parameters
    H = np.append(np.eye(p), np.zeros((p, n)), axis=axis)[:, 0:n]
    R = variance_obs_comp * np.eye(p)

    return H, R


def input_matrices_M_Q(
    states: np.ndarray,
    variance_state_comp: float = 0.0001,
    axis: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create default input observation matrices M and Q for a Kalman algorithms.

    H will only contain values of 1 at corresponding positions

    Parameters:
        states (ndarray) shape(T,n): Array of state variables.
        variance_state_comp (float): Variance of state component. Default to 0.0001.
        axis (float): axis along which the number of variables is defined. Default to 1.

    Returns:
        Tuple: A Tuple containing the input matrices (M, Q).
        - M : np.ndarray, shape (n, n)
            Model matrix that maps the state space to the next timestep.
        - Q : np.ndarray, shape (n, n)
            Model noise covariance matrix.
    """
    # shapes
    n = np.shape(states)[axis]

    # kalman parameters
    M = np.eye(n)
    Q = variance_state_comp * np.eye(n)

    return M, Q


# Kalman filter


def xarray_Kalman_filter(
    ds: xr.Dataset,
    state_variables: Iterable[str],
    observation_variables: Iterable[str],
    initial_covariance_matrix: np.ndarray,
    M: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    dim="time",
    estimation_idx=0,
    suffix: str = "",
) -> xr.Dataset:
    """
    NOTE: !!!!!!!!!! NOT READY !!!!!!!!!!
    Run the Kalman filter on the input dataset and return the results in an xarray Dataset.

    This function applies the Kalman filter algorithm on the specified observations and state variables in the given xarray dataset.
    The results are stored in a new xarray dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing the state and state variables.
    state_variables : list len(n)
        List of state variables to be used in the Kalman filter.
    observation_variables : list len(p)
        List of observed variables to be used in the Kalman filter.
    initial_covariance_matrix : np.ndarray shape(n,n)
        Initial Covarince Matrix corresponding to the initial estimation of the state variables.
    M : array-like, shape (n, n)
        Transition matrix.
    Q : array-like, shape (T, n, n)
        Process noise covariance matrix.
    H : array-like, shape (p, n)
        Observation matrix that maps the state space to the observation space.
    R : array-like, shape (p, p)
        Covariance matrix representing the observation noise.
    dim : str
        Dimension along which the estimation state shall be selected.
        Default dimension is "time".
    estimation_idx : int
        Index used for the inital estimation of the state from along 'dim'.
        Defaul to 0.
    suffix : str, optional
        Suffix to be appended to the variable names in the output dataset.
        Default is an empty string.

    Returns
    -------
    xarray.Dataset
        New xarray dataset containing the results of the Kalman filter.
        Dimensions are:

    Output Dimensions:
    - time: The time dimension of the input dataset.
    - state_name: Dimension representing the state variables.
    - state_name_copy: Dimension representing a copy of the state variables.

    Output Coordinates:
    - time: Coordinates corresponding to the time of the input dataset.
    - state_name: Coordinates corresponding to the state variables.
    - state_name_copy: Coordinates corresponding to the copy of the state variables.

    Output Variables:
    - state_forecast<suffix>: DataArray containing the estimated states over time.
        - Dimensions: time, state_name
    - covariance_forecast<suffix>: DataArray containing the covariance of the estimated states over time.
        - Dimensions: time, state_name, state_name_copy
    - state_assimilation<suffix>: DataArray containing the estimated states over time.
        - Dimensions: time, state_name
    - covariance_assimilation<suffix>: DataArray containing the covariance of the estimated states over time.
        - Dimensions: time, state_name, state_name_copy
    - log_likelihod<suffix>: DataArray containing the log-likelihood values for each timestep.
        - Dimensions: time
    # Not yet implemented.
    # - kalman_gain<suffix>: DataArray containing the Kalman gain values for each timestep.
    #     - Dimensions: time

    Note:
    - The output variable names will be suffixed with the provided suffix parameter.
    - If your given xr.Dataset was provided using a selection by values or indices, it is suggested to use the expand_and_assign_coords() function in order to contain the correct values of the dimensions and coordinates.
    """
    # raise NotImplementedError("Function full deployed yet!")
    # function to craete new names from a list of strings
    join_names = lambda l: "".join(l)

    n = len(state_variables)
    p = len(observation_variables)

    # check the dimensions:
    assert np.shape(initial_covariance_matrix) == (
        n,
        n,
    ), f"Mismatch in dimensions of initial covariance matrix : {np.shape(initial_covariance_matrix)} but should be {(n,n)}"
    assert np.shape(M) == (
        n,
        n,
    ), f"Mismatch in dimensions of R : {np.shape(M)} but should be {(n,n)}"
    assert np.shape(Q) == (
        n,
        n,
    ), f"Mismatch in dimensions of Q : {np.shape(Q)} but should be {(n,n)}"
    assert np.shape(H) == (
        p,
        n,
    ), f"Mismatch in dimensions of R : {np.shape(H)} but should be {(p,n)}"
    assert np.shape(R) == (
        p,
        p,
    ), f"Mismatch in dimensions of R : {np.shape(R)} but should be {(p,p)}"

    # create a list of numpy array to be used for the Kalman itteration
    observation_list = []
    state_list = []
    for var in observation_variables:
        observation_list.append(ds[var].values.flatten())
    for var in state_variables:
        state_list.append(ds[var].isel({dim: estimation_idx}).values.flatten())

    initial_state_estimation, observations = input_arrays(
        observation_list=observation_list, state_list=state_list
    )

    # initial_state_estimation has shape (1,n) but we need it to be (n)
    initial_state_estimation = initial_state_estimation[0]

    # ---------------
    # run the Kalman_SEM algorithm
    # ---------------
    (x_f, P_f, x_a, P_a, log_likelihod, K_a) = Kalman_filter(
        y=observations,
        x0=initial_state_estimation,
        P0=initial_covariance_matrix,
        M=M,
        Q=Q,
        H=H,
        R=R,
    )

    # ---------------
    # create result dataset in which to store the whole lists
    # ---------------
    result = xr.Dataset({})
    # assign coordinates
    result = result.assign_coords(
        dict(
            time=ds["time"],
            state_name=state_variables,
            state_name_copy=state_variables,
            observation_name=observation_variables,
        )
    )
    # store x_s
    new_var = join_names(["state_forecast", suffix])
    result[new_var] = xr.DataArray(
        data=x_f,
        dims=["time", "state_name"],
    )
    # store P_s
    new_var = join_names(["covariance_forecast", suffix])
    result[new_var] = xr.DataArray(
        data=P_f,
        dims=["time", "state_name", "state_name_copy"],
    )
    # store x_a
    new_var = join_names(["state_assimilation", suffix])
    result[new_var] = xr.DataArray(
        data=x_a,
        dims=["time", "state_name"],
    )
    # store P_a
    new_var = join_names(["covariance_assimilation", suffix])
    result[new_var] = xr.DataArray(
        data=P_a,
        dims=["time", "state_name", "state_name_copy"],
    )
    # store the log_likelihod
    new_var = join_names(["log_likelihod", suffix])
    result[new_var] = xr.DataArray(
        data=log_likelihod,
        dims=["time"],
    )
    # store Kalman_gain:
    # TODO: Not NotImplementedError

    return result


# Kalman smoother


def xarray_Kalman_smoother(
    ds: xr.Dataset,
    state_variables: Iterable[str],
    observation_variables: Iterable[str],
    initial_covariance_matrix: np.ndarray,
    M: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    dim="time",
    estimation_idx=0,
    suffix: str = "",
) -> xr.Dataset:
    """
    NOTE: !!!!!!!!!! NOT READY !!!!!!!!!!
    Run the Kalman smoother on the input dataset and return the results in an xarray Dataset.

    This function applies the Kalman smoother algorithm on the specified observations and state variables in the given xarray dataset.
    The results are stored in a new xarray dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing the state and state variables.
    state_variables : list len(n)
        List of state variables to be used.
    observation_variables : list len(p)
        List of observed variables to be used.
    initial_covariance_matrix : np.ndarray shape(n,n)
        Initial Covarince Matrix corresponding to the initial estimation of the state variables.
    M : array-like, shape (n, n)
        Transition matrix.
    Q : array-like, shape (T, n, n)
        Process noise covariance matrix.
    H : array-like, shape (p, n)
        Observation matrix that maps the state space to the observation space.
    R : array-like, shape (p, p)
        Covariance matrix representing the observation noise.
    dim : str
        Dimension along which the estimation state shall be selected.
        Default dimension is "time".
    estimation_idx : int
        Index used for the inital estimation of the state from along 'dim'.
        Defaul to 0.
    suffix : str, optional
        Suffix to be appended to the variable names in the output dataset.
        Default is an empty string.

    Returns
    -------
    xarray.Dataset
        New xarray dataset containing the results of the Kalman smoother.

    Output Dimensions:
    - time: The time dimension of the input dataset.
    - state_name: Dimension representing the state variables.
    - state_name_copy: Dimension representing a copy of the state variables.

    Output Coordinates:
    - time: Coordinates corresponding to the time of the input dataset.
    - state_name: Coordinates corresponding to the state variables.
    - state_name_copy: Coordinates corresponding to the copy of the state variables.

    Output Variables:
    - state_forecast<suffix>: DataArray containing the estimated states over time.
        - Dimensions: time, state_name
    - covariance_forecast<suffix>: DataArray containing the covariance of the estimated states over time.
        - Dimensions: time, state_name, state_name_copy
    - state_assimilation<suffix>: DataArray containing the estimated states over time.
        - Dimensions: time, state_name
    - covariance_assimilation<suffix>: DataArray containing the covariance of the estimated states over time.
        - Dimensions: time, state_name, state_name_copy
    - state_smooth<suffix>: DataArray containing the estimated states over time.
        - Dimensions: time, state_name
    - covariance_smooth<suffix>: DataArray containing the covariance of the estimated states over time.
        - Dimensions: time, state_name, state_name_copy
    - log_likelihod<suffix>: DataArray containing the log-likelihood values for each timestep.
        - Dimensions: time
    # Not yet implemented.
    # - kalman_gain<suffix>: DataArray containing the Kalman gain values for each timestep.
    #     - Dimensions: time

    Note:
    - The output variable names will be suffixed with the provided suffix parameter.
    - If your given xr.Dataset was provided using a selection by values or indices, it is suggested to use the expand_and_assign_coords() function in order to contain the correct values of the dimensions and coordinates.
    """
    # raise NotImplementedError("Function full deployed yet!")
    # function to craete new names from a list of strings
    join_names = lambda l: "".join(l)

    n = len(state_variables)
    p = len(observation_variables)

    # check the dimensions:
    assert np.shape(initial_covariance_matrix) == (
        n,
        n,
    ), f"Mismatch in dimensions of initial covariance matrix : {np.shape(initial_covariance_matrix)} but should be {(n,n)}"
    assert np.shape(M) == (
        n,
        n,
    ), f"Mismatch in dimensions of R : {np.shape(M)} but should be {(n,n)}"
    assert np.shape(Q) == (
        n,
        n,
    ), f"Mismatch in dimensions of Q : {np.shape(Q)} but should be {(n,n)}"
    assert np.shape(H) == (
        p,
        n,
    ), f"Mismatch in dimensions of R : {np.shape(H)} but should be {(p,n)}"
    assert np.shape(R) == (
        p,
        p,
    ), f"Mismatch in dimensions of R : {np.shape(R)} but should be {(p,p)}"

    # create a list of numpy array to be used for the Kalman itteration
    observation_list = []
    state_list = []
    for var in observation_variables:
        observation_list.append(ds[var].values.flatten())
    for var in state_variables:
        state_list.append(ds[var].isel({dim: estimation_idx}).values.flatten())

    initial_state_estimation, observations = input_arrays(
        observation_list=observation_list, state_list=state_list
    )

    # initial_state_estimation has shape (1,n) but we need it to be (n)
    initial_state_estimation = initial_state_estimation[0]

    # ---------------
    # run the Kalman_SEM algorithm
    # ---------------
    (x_f, P_f, x_a, P_a, x_s, P_s, log_likelihod, P_s_lag) = Kalman_smoother(
        y=observations,
        x0=initial_state_estimation,
        P0=initial_covariance_matrix,
        M=M,
        Q=Q,
        H=H,
        R=R,
    )

    # ---------------
    # create result dataset in which to store the whole lists
    # ---------------
    result = xr.Dataset({})
    # assign coordinates
    result = result.assign_coords(
        dict(
            time=ds["time"],
            state_name=state_variables,
            state_name_copy=state_variables,
            observation_name=observation_variables,
        )
    )
    # store x_s
    new_var = join_names(["state_forecast", suffix])
    result[new_var] = xr.DataArray(
        data=x_f,
        dims=["time", "state_name"],
    )
    # store P_s
    new_var = join_names(["covariance_forecast", suffix])
    result[new_var] = xr.DataArray(
        data=P_f,
        dims=["time", "state_name", "state_name_copy"],
    )
    # store x_a
    new_var = join_names(["state_assimilation", suffix])
    result[new_var] = xr.DataArray(
        data=x_a,
        dims=["time", "state_name"],
    )
    # store P_a
    new_var = join_names(["covariance_assimilation", suffix])
    result[new_var] = xr.DataArray(
        data=P_a,
        dims=["time", "state_name", "state_name_copy"],
    )
    # store x_s
    new_var = join_names(["state_smooth", suffix])
    result[new_var] = xr.DataArray(
        data=x_s,
        dims=["time", "state_name"],
    )
    # store P_s
    new_var = join_names(["covariance_smooth", suffix])
    result[new_var] = xr.DataArray(
        data=P_s,
        dims=["time", "state_name", "state_name_copy"],
    )
    # store the log_likelihod
    new_var = join_names(["log_likelihod", suffix])
    result[new_var] = xr.DataArray(
        data=log_likelihod,
        dims=["time"],
    )
    # store Kalman_gain:
    # TODO: Not NotImplementedError

    return result


# Kalman SEM


def xarray_Kalman_SEM(
    ds: xr.Dataset,
    observation_variables: Iterable[str],
    state_variables: Iterable[str],
    nb_iter_SEM: int = 30,
    variance_obs_comp: float = 0.0001,
    suffix: str = "",
) -> xr.Dataset:
    """
    Run the Kalman SEM algorithm on the input dataset and return the results in
    an xarray Dataset.

    This function applies the Kalman_SEM algorithm on the specified observations and state variables in the given xarray dataset.
    It performs a specified number of iterations and computes the state estimates, covariance, transition matrix, and
    process noise covariance matrix. The results are stored in a new xarray dataset.

    Note:
    - The state_varianbles can contain latent variables which are not found in the observation_variables

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing the state and state variables.
    observation_variables : list len(p)
        List of observed variables to be used in the Kalman SEM algorithm.
    state_variables : list len(n)
        List of state variables to be used in the Kalman SEM algorithm.
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
    - state_name: Dimension representing the state variables.
    - state_name_copy: Dimension representing a copy of the state variables.

    Output Coordinates:
    - time: Coordinates corresponding to the time of the input dataset.
    - state_name: Coordinates corresponding to the state variables.
    - state_name_copy: Coordinates corresponding to the copy of the state variables.
    - kalman_itteration: Coordinates representing the Kalman SEM iteration index.

    Output Variables:
    - states<suffix>: DataArray containing the estimated states over time.
        - Dimensions: time, state_name
    - covariance<suffix>: DataArray containing the covariance of the estimated states over time.
        - Dimensions: time, state_name, state_name_copy
    - M<suffix>: DataArray containing the transition matrix M.
        - Dimensions: state_name, state_name_copy
    - Q<suffix>: DataArray containing the observation noise covariance matrix Q.
        - Dimensions: state_name, state_name_copy
    - log_likelihod<suffix>: DataArray containing the log-likelihood values for each Kalman SEM iteration.
        - Dimensions: kalman_itteration

    Note:
    - The output variable names will be suffixed with the provided suffix parameter.
    - If your given xr.Dataset was provided using a selection by values or indices, it is suggested to use the expand_and_assign_coords() function in order to contain the correct values of the dimensions and coordinates.
    """

    # function to craete new names from a list of strings
    join_names = lambda l: "".join(l)

    # make sure that the variables in observation_variables are in same order as in state_variables
    state_variables = ordered_like(state_variables, observation_variables)
    # check that state_variables is a subset sorted like observation_variables
    assert_ordered_subset(observation_variables, state_variables)

    # create a list of numpy array to be used for the Kalman itteration
    observation_list = []
    state_list = []
    for var in observation_variables:
        observation_list.append(ds[var].values.flatten())
    for var in state_variables:
        state_list.append(ds[var].values.flatten())

    states, observations = input_arrays(
        state_list=state_list, observation_list=observation_list
    )

    H, R = input_matrices_H_R(
        states=states, observations=observations, variance_obs_comp=variance_obs_comp
    )
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
    ) = Kalman_SEM(x=states, y=observations, H=H, R=R, nb_iter_SEM=nb_iter_SEM)

    # ---------------
    # create result dataset in which to store the whole lists
    # ---------------
    result = xr.Dataset({})
    # assign coordinates
    result = result.assign_coords(
        dict(
            time=ds["time"],
            state_name=state_variables,
            state_name_copy=state_variables,
            kalman_itteration=np.arange(nb_iter_SEM),
        )
    )
    # store x_s
    new_var = join_names(["states", suffix])
    result[new_var] = xr.DataArray(
        data=x_s,
        dims=["time", "state_name"],
    )
    # store P_s
    new_var = join_names(["covariance", suffix])
    result[new_var] = xr.DataArray(
        data=P_s,
        dims=["time", "state_name", "state_name_copy"],
    )
    # store M
    new_var = join_names(["M", suffix])
    result[new_var] = xr.DataArray(
        data=M,
        dims=["state_name", "state_name_copy"],
    )
    # store Q
    new_var = join_names(["Q", suffix])
    result[new_var] = xr.DataArray(
        data=Q,
        dims=["state_name", "state_name_copy"],
    )
    # store the log_likelihod
    new_var = join_names(["log_likelihod", suffix])
    result[new_var] = xr.DataArray(
        data=log_likelihod,
        dims=["kalman_itteration"],
    )

    return result


def xarray_Kalman_SEM_time_dependent(
    ds: xr.Dataset,
    observation_variables: Iterable[str],
    state_variables: Iterable[str],
    nb_iter_SEM: int = 30,
    variance_obs_comp: float = 0.0001,
    sigma=10,
    suffix: str = "",
) -> xr.Dataset:
    """
    Run the time dependent Kalman SEM algorithm on the input dataset and return
    the results in an xarray Dataset.

    This function applies the Kalman_SEM_time_dependent algorithm on the specified observations and state variables in the given xarray dataset.
    It performs a specified number of iterations and computes the state estimates, covariance, transition matrix, and
    process noise covariance matrix. The results are stored in a new xarray dataset.

    Note:
    - The state_varianbles can contain latent variables which are not found in the observation_variables

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing the state and state variables.
    observation_variables : list len(p)
        List of observed variables to be used in the Kalman SEM algorithm.
    state_variables : list len(n)
        List of state variables to be used in the Kalman SEM algorithm.
    nb_iter_SEM : int, optional
        Number of iterations for the Kalman SEM algorithm.
        Default is 30.
    variance_obs_comp : float, optional
        Variance parameter for observation components in the Kalman SEM algorithm.
        Default is 0.0001.
    sigma : float
        Standard deviation as the sqrt(variance) of the Gaussian distribution to create the 1D kernel used for the local linear regression.
        Note that the sigma is unitless and is measurent in index-positions of the array.
    suffix : str, optional
        Suffix to be appended to the variable names in the output dataset.
        Default is an empty string.

    Returns
    -------
    xarray.Dataset
        New xarray dataset containing the results of the Kalman_SEM algorithm.
        Dimensions are:

    Output Dimensions:
    - time: The time dimension of the input dataset. shape(t)
    - state_name: Dimension representing the state variables. shape(n)
    - state_name_copy: Dimension representing a copy of the state variables. shape(n)

    Output Coordinates:
    - time: Coordinates corresponding to the time of the input dataset.
    - state_name: Coordinates corresponding to the state variables.
    - state_name_copy: Coordinates corresponding to the copy of the state variables.
    - kalman_itteration: Coordinates representing the Kalman SEM iteration index.

    Output Variables:
    - states<suffix>: DataArray containing the estimated states over time.
        - Dimensions: time, state_name
    - covariance<suffix>: DataArray containing the covariance of the estimated states over time.
        - Dimensions: time, state_name, state_name_copy
    - M<suffix>: DataArray containing the transition matrix M.
        - Dimensions: state_name, state_name_copy
    - Q<suffix>: DataArray containing the observation noise covariance matrix Q.
        - Dimensions: state_name, state_name_copy
    - log_likelihod<suffix>: DataArray containing the log-likelihood values for each Kalman SEM iteration.
        - Dimensions: kalman_itteration

    Note:
    - The output variable names will be suffixed with the provided suffix parameter.
    - If your given xr.Dataset was provided using a selection by values or indices, it is suggested to use the expand_and_assign_coords() function in order to contain the correct values of the dimensions and coordinates.
    """

    # function to craete new names from a list of strings
    join_names = lambda l: "".join(l)

    # make sure that the variables in observation_variables are in same order as in state_variables
    state_variables = ordered_like(state_variables, observation_variables)
    # check that state_variables is a subset sorted like observation_variables
    assert_ordered_subset(observation_variables, state_variables)

    # create a list of numpy array to be used for the Kalman itteration
    observation_list = []
    state_list = []
    for var in observation_variables:
        observation_list.append(ds[var].values.flatten())
    for var in state_variables:
        state_list.append(ds[var].values.flatten())

    states, observations = input_arrays(
        state_list=state_list, observation_list=observation_list
    )
    H, R = input_matrices_H_R(
        states=states, observations=observations, variance_obs_comp=variance_obs_comp
    )
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
    ) = Kalman_SEM_time_dependent(
        x=states,
        y=observations,
        H=H,
        R=R,
        nb_iter_SEM=nb_iter_SEM,
        sigma=sigma,
    )

    # ---------------
    # create result dataset in which to store the whole lists
    # ---------------
    result = xr.Dataset({})
    # assign coordinates
    result = result.assign_coords(
        dict(
            time=ds["time"],
            state_name=state_variables,
            state_name_copy=state_variables,
            kalman_itteration=np.arange(nb_iter_SEM),
        )
    )
    # store x_s
    new_var = join_names(["states", suffix])
    result[new_var] = xr.DataArray(
        data=x_s,
        dims=["time", "state_name"],
    )
    # store P_s
    new_var = join_names(["covariance", suffix])
    result[new_var] = xr.DataArray(
        data=P_s,
        dims=["time", "state_name", "state_name_copy"],
    )
    # store M
    new_var = join_names(["M", suffix])
    result[new_var] = xr.DataArray(
        data=M,
        dims=["time", "state_name", "state_name_copy"],
    )
    # store Q
    new_var = join_names(["Q", suffix])
    result[new_var] = xr.DataArray(
        data=Q,
        dims=["time", "state_name", "state_name_copy"],
    )
    # store the log_likelihod
    new_var = join_names(["log_likelihod", suffix])
    result[new_var] = xr.DataArray(
        data=log_likelihod,
        dims=["kalman_itteration"],
    )

    return result


# Functions to handle dimensional or coordinate problems:


def expand_and_assign_coords(
    ds1: xr.Dataset, ds2: xr.Dataset, select_dict: Dict = dict()
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
    >>> result = expand_and_assign_coords(ds1, ds2, select_dict)
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


# Functions to add one ore more varibales to a DataSet


def add_random_variable(
    ds: xr.Dataset,
    var_name: str,
    random_generator: np.random.Generator,
    variance: float,
    dim: str = "time",
) -> None:
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


def create_empty_dataarray(ds1: xr.Dataset, ds2: xr.Dataset) -> xr.DataArray:
    """
    Create an empty DataArray by combining the coordinates from two given
    datasets or dataarrays.

    Parameters:
        ds1 (xarray.Dataset or xarray.DataArray): The first dataset.
        ds2 (xarray.Dataset or xarray.DataArray): The second dataset.

    Returns:
        xarray.DataArray: An empty DataArray with coordinates from both ds1 and ds2.
    """
    coords = {**ds1.coords, **ds2.coords}
    return xr.DataArray(coords=coords.values(), dims=coords.keys())


def add_empty_dataarrays(ds1: xr.Dataset, ds2: xr.Dataset, new_dimension: str) -> None:
    """
    Add empty data arrays to ds1 for each variable in ds2.

    This function adds empty data arrays to ds1 for each variable in ds2 using the `empty_dataarray_from_two` function.
    The empty data arrays are constructed based on the dimensions of ds1's new_dimension and ds2's variables.

    Parameters:
        ds1 (xr.Dataset): The target dataset where empty data arrays will be added.
        ds2 (xr.Dataset): The source dataset from which variables will be used to construct the empty data arrays.
        new_dimension (str): The name of the new dimension.

    Returns:
        None

    Example:
    >>> ds1 = xr.Dataset(
        {"var1": (("x", "y"), np.zeros((3, 3)))},
        coords={"x": [1, 2, 3], "y": [4, 5, 6]}
    )
    >>> ds2 = xr.Dataset(
        {"var2": (("x", "y"), np.ones((3, 3)))},
        coords={"x": [2, 3, 4], "y": [5, 6, 7]}
    )
    >>> new_dimension = "run"
    >>> add_all_empty_dataarray(ds1, ds2, new_dimension)
    >>> print(ds1)
    <xarray.Dataset>
    Dimensions:  (x: 3, y: 3, run: 1)
    Coordinates:
      * x        (x) int64 1 2 3
      * y        (y) int64 4 5 6
      * run      (run) int64 0
    Data variables:
        var1     (x, y) float64 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
        var2     (run, x, y) float64 nan nan nan nan nan nan nan nan nan
    """

    # Iterate over all variables in ds2
    for var in ds2.data_vars:
        # Create an empty data array using empty_dataarray_from_two function
        empty_dataarray = create_empty_dataarray(ds1[new_dimension], ds2[var])
        # Assign the empty data array to the corresponding variable in ds1
        ds1[var] = empty_dataarray


def assign_variables_by_selection(
    ds1: xr.Dataset, ds2: xr.Dataset, select_dict: Dict = dict()
) -> None:
    """
    Set all variables from ds2 into ds1 at the specified selection coordinates.

    This function assigns the values of all variables from ds2 to the corresponding variables in ds1
    at the specified selection coordinates defined by select_dict.

    Parameters:
        ds1 (xr.Dataset): The target dataset where variables will be set.
        ds2 (xr.Dataset): The source dataset from which variables will be taken.
        select_dict (dict): Dictionary of selection or index selection containing names and values of the coordinates to select.

    Returns:
        None

    Example:
    >>> ds1 = xr.Dataset(
        {"var1": (("x", "y"), np.zeros((3, 3))), "var2": (("x", "y"), np.ones((3, 3)))},
        coords={"x": [1, 2, 3], "y": [4, 5, 6]}
    )
    >>> ds2 = xr.Dataset(
        {"var1": (("x", "y"), np.full((3, 3), 2)), "var2": (("x", "y"), np.full((3, 3), 3))},
        coords={"x": [2, 3, 4], "y": [5, 6, 7]}
    )
    >>> select_dict = {"x": 2, "y": 6}
    >>> set_all_variables_for_selection(ds1, ds2, select_dict)
    >>> print(ds1)
    <xarray.Dataset>
    Dimensions:  (x: 3, y: 3)
    Coordinates:
      * x        (x) int64 1 2 3
      * y        (y) int64 4 5 6
    Data variables:
        var1     (x, y) float64 0.0 0.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0
        var2     (x, y) float64 1.0 1.0 1.0 1.0 1.0 3.0 1.0 1.0 1.0
    """

    # Iterate over all variables in ds2
    for var in ds2.data_vars:
        # Assign values from ds2 to corresponding variables in ds1 at the selection coordinates
        ds1[var].loc[select_dict] = ds2[var].loc[select_dict]


def assign_variable_by_double_selection(
    ds1: xr.Dataset,
    da2: xr.DataArray,
    var_name: str,
    select_dict1: Dict = dict(),
    select_dict2: Dict = dict(),
) -> None:
    """
    Set variables from DataArray da2 into ds1 at the specified selection
    coordinates.

    This function assigns the values of the variables from da2 to the corresponding variables in ds1
    at the specified selection coordinates defined by select_dict.

    Parameters:
        ds1 (xr.Dataset): The target dataset where variables will be set.
        da2 (xr.DataArray): The source dataset from which variables will be taken.
        var_name (str) : variable name into which da2 shall be inserted.
        select_dict1 (dict): Dictionary of selection or index selection containing names and values of the coordinates to select from ds1.
        select_dict2 (dict): Dictionary of selection or index selection containing names and values of the coordinates to select from da2.

    Returns:
        None
    """
    # Iterate over all variables in ds2
    # Assign values from ds2 to corresponding variables in ds1 at the selection coordinates
    ds1[var_name].loc[select_dict1] = da2.loc[select_dict2]


def assign_variables_by_double_selection(
    ds1: xr.Dataset,
    ds2: xr.Dataset,
    select_dict1: Dict = dict(),
    select_dict2: Dict = dict(),
) -> None:
    """
    Set all variables from ds2 into ds1 at the specified selection coordinates
    for both datasets.

    This function assigns the values of all variables from ds2 to the corresponding variables in ds1
    at the specified selection coordinates defined by select_dict.

    Parameters:
        ds1 (xr.Dataset): The target dataset where variables will be set.
        ds2 (xr.Dataset): The source dataset from which variables will be taken.
        select_dict1 (dict): Dictionary of selection or index selection containing names and values of the coordinates to select from ds1.
        select_dict2 (dict): Dictionary of selection or index selection containing names and values of the coordinates to select from ds2.

    Returns:
        None

    Example:
    >>> ds1 = xr.Dataset(
        {"var1": (("x", "y"), np.zeros((3, 3))), "var2": (("x", "y"), np.ones((3, 3)))},
        coords={"x": [1, 2, 3], "y": [4, 5, 6]}
    )
    >>> ds2 = xr.Dataset(
        {"var1": (("x", "y"), np.full((3, 3), 2)), "var2": (("x", "y"), np.full((3, 3), 3))},
        coords={"x": [2, 3, 4], "y": [5, 6, 7]}
    )
    >>> select_dict = {"x": 2, "y": 6}
    >>> set_all_variables_for_selection(ds1, ds2, select_dict)
    >>> print(ds1)
    <xarray.Dataset>
    Dimensions:  (x: 3, y: 3)
    Coordinates:
      * x        (x) int64 1 2 3
      * y        (y) int64 4 5 6
    Data variables:
        var1     (x, y) float64 0.0 0.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0
        var2     (x, y) float64 1.0 1.0 1.0 1.0 1.0 3.0 1.0 1.0 1.0
    """

    # Iterate over all variables in ds2
    for var in ds2.data_vars:
        # Assign values from ds2 to corresponding variables in ds1 at the selection coordinates
        ds1[var].loc[select_dict1] = ds2[var].loc[select_dict2]


#  Functions for Experiments or analysis


def multiple_runs_of_func(
    ds: xr.Dataset,
    func: Callable[[xr.Dataset, Dict], xr.Dataset],
    func_kwargs: Dict,
    number_of_runs: int = 2,
    new_dimension: str = "run",
) -> xr.Dataset:
    """
    Run a function multiple times on a dataset and combine the results into an
    xarray.Dataset.

    Parameters:
        ds (xarray.Dataset): The input dataset.
        func (callable): The function to be executed on the dataset.
        func_kwargs (dict): Keyword arguments to be passed to the function.
        number_of_runs (int): The number of times to run the function. Default is 2.
        new_dimension (str): The name of the new dimension to be added for each run. Default is "run".

    Returns:
        xarray.Dataset: A dataset containing the results of running the function multiple times.

    Notes:
        - The function `func` should accept the dataset `ds` as the first argument and the keyword
          arguments `func_kwargs`.
        - The function `func` should return an xarray.Dataset.
    """
    # Create an empty dataset to store the results
    result = xr.Dataset({})

    # Assign a new dimension to the dataset
    result = result.assign_coords({new_dimension: np.arange(number_of_runs)})

    # for the first itteration use the result to create the new coordinates and varibles needed in the output DataSet
    current_run = 0
    select_dict = {new_dimension: current_run}
    # Execute the function on the dataset
    func_result = func(ds=ds, **func_kwargs)
    # For the first run, expand the dimensions of the result dataset
    result = expand_and_assign_coords(ds1=result, ds2=func_result, select_dict={})
    # Create empty data arrays in the result dataset for each variable in the function result
    add_empty_dataarrays(ds1=result, ds2=func_result, new_dimension=new_dimension)

    for current_run in range(1, number_of_runs):
        # Create a dictionary to select the current run
        select_dict = {new_dimension: current_run}

        # Execute the function on the dataset
        func_result = func(ds=ds, **func_kwargs)

        # Expand the dimensions of the function result dataset to match the result dataset
        func_result = expand_and_assign_coords(
            ds1=func_result, ds2=result, select_dict=select_dict
        )

        # Assign the values from the function result to the result dataset
        assign_variables_by_selection(
            ds1=result, ds2=func_result, select_dict=select_dict
        )

    return result
