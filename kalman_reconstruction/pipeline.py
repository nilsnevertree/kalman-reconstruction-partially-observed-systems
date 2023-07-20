import traceback

from typing import Callable, Dict, Iterable, Tuple, Union
from warnings import warn

import numpy as np
import xarray as xr

from kalman_reconstruction.kalman import (
    Kalman_filter,
    Kalman_SEM,
    Kalman_SEM_full_output,
    Kalman_smoother,
)
from kalman_reconstruction.kalman_time_dependent import (
    Kalman_filter_time_dependent,
    Kalman_SEM_time_dependent,
    Kalman_smoother_time_dependent,
)
from kalman_reconstruction.statistics import (
    assert_ordered_subset,
    kalman_single_forecast,
    ordered_like,
)


# Kalman_Functions


def input_arrays_combined(
    observation_list: Iterable[np.ndarray], random_list: Iterable[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input arrays of observations (y) and states (x) for the
    Kalman_algorithms. The state is a concatenation of the observation
    variables followed by the random variables.

    Notes:
        - All arrays in the list must be 1D and of equal length T.

    Parameters:
        observation_list (list) len(p): List of observation variables.
        random_list (list) len(r): List of latent variables that shall be appended.

    Returns:
        Tuple : A Tuple containing the input arrays for the Kalman_algorithms.
            - states : np.ndarray shape(T, p+r),
            - observations : np.ndarray shape(T, p)

    Examples:
        >>> observation_list = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
        >>> random_list = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        >>> states, observations = input_arrays_combined(observation_list, random_list)

        >>> # The 'states' thus have shape (T, p+r)
        >>> print(states)
        [[1.  4.  7.  0.1 0.4]
         [2.  5.  8.  0.2 0.5]
         [3.  6.  9.  0.3 0.6]]

        >>> # The 'observations' thus have shape (T, p+r)
        >>> print(observations)
        [[1 4 7]
         [2 5 8]
         [3 6 9]]
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

    Notes:
        - All arrays in the list must be 1D and of equal length T.

    Parameters:
        state_list (list) len(n): List of state variables.
        observation_list (list) len(p): List of observed variables.

    Returns:
        Tuple : A Tuple containing the input arrays for the Kalman_algorithms.
            - states : np.ndarray shape(T, n),
            - observations : np.ndarray shape(T, p)

    Examples:
        >>> state_list = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        >>> observation_list = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
        >>> states, observations = input_arrays(state_list, observation_list)
        >>> print(states)
        [[0.1 0.4]
         [0.2 0.5]
         [0.3 0.6]]
        >>> print(observations)
        [[1 4 7]
         [2 5 8]
         [3 6 9]]
    """
    observations = np.array(observation_list).T
    states = np.array(state_list).T
    assert np.all(
        observations[1:].shape == observations[:-1].shape
    ), "Not all arrays in 'observations' are of the same shape!"
    assert np.all(
        states[1:].shape == states[:-1].shape
    ), "Not all arrays in 'states' are of the same shape!"
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
    Examples:
        >>> states = np.array(
            [[1.0, 2.0, 3.0],
             [3.0, 4.0, 5.0],
             [5.0, 6.0, 7.0]])
        >>> observations = np.array(
            [[10.0, 20.0],
             [30.0, 40.0],
             [50.0, 60.0]])
        >>> variance_obs_comp = 0.4
        >>> H, R = input_matrices_H_R(states, observations, variance_obs_comp, axis=1)
        >>> print(H)
        [[1. 0. 0.]
        [0. 1. 0.]]
        >>> print(R)
        [[0.4 0. ]
        [0.  0.4]]
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

    Examples:
        >>> states = np.array(
        ... [[1.0, 2.0],
        ... [3.0, 4.0],
        ... [5.0, 6.0]]
        ... )
        >>> variance_state_comp = 0.01
        >>> M, Q = input_matrices_M_Q(states, variance_state_comp, axis=1)
        >>> print(M)
        [[1. 0.]
         [0. 1.]]
        >>> print(Q)
        [[0.01 0.  ]
         [0.   0.01]]
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
    suffix: str = "",
) -> xr.Dataset:
    """
    Notes: !!!!!!!!!! NOT READY !!!!!!!!!!
    Run the Kalman filter on the input dataset and return the results in an xarray Dataset.

    This function applies the Kalman filter algorithm on the specified observations and state variables in the given xarray dataset.
    The results are stored in a new xarray dataset.

    Notes
    -----
        - The output variable names will be suffixed with the provided suffix parameter.
        - If your given xr.Dataset was provided using a selection by values or indices, it is suggested to use the expand_and_assign_coords() function in order to contain the correct values of the dimensions and coordinates.

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
    Q : array-like, shape (n, n)
        Process noise covariance matrix.
    H : array-like, shape (p, n)
        Observation matrix that maps the state space to the observation space.
    R : array-like, shape (p, p)
        Covariance matrix representing the observation noise.
    dim : str
        Dimension along which the estimation state shall be selected.
        Default dimension is "time".
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
            - state_forecast<suffix>:
                - DataArray containing the estimated states over time.
                - Dimensions: time, state_name
            - covariance_forecast<suffix>:
                - DataArray containing the covariance of the estimated states over time.
                - Dimensions: time, state_name, state_name_copy
            - state_assimilation<suffix>:
                - DataArray containing the estimated states over time.
                - Dimensions: time, state_name
            - covariance_assimilation<suffix>:
                - DataArray containing the covariance of the estimated states over time.
                - Dimensions: time, state_name, state_name_copy
            - log_likelihod<suffix>:
                - DataArray containing the log-likelihood values for each timestep.
                - Dimensions: time
            # Not yet implemented.
            # - kalman_gain<suffix>: DataArray containing the Kalman gain values for each timestep.
            #     - Dimensions: time
    """
    # raise NotImplementedError("Function full deployed yet!")
    # function to create new names from a list of strings
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

    # create a list of numpy array to be used for the Kalman iteration
    observation_list = []
    state_list = []
    for var in observation_variables:
        observation_list.append(ds[var].values.flatten())
    # for var in state_variables:
    #     state_list.append(ds[var].isel({dim: estimation_idx}).values.flatten())

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


def xarray_Kalman_filter_time_dependent(
    ds: xr.Dataset,
    state_variables: Iterable[str],
    observation_variables: Iterable[str],
    initial_state_estimation: np.ndarray,
    initial_covariance_matrix: np.ndarray,
    M: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    dim="time",
    suffix: str = "",
) -> xr.Dataset:
    """
    Notes: !!!!!!!!!! NOT READY !!!!!!!!!!
    Run the Kalman filter in the time dependent version on the input dataset and return the results in an xarray Dataset.

    This function applies the Kalman filter algorithm on the specified observations and state variables in the given xarray dataset.
    The results are stored in a new xarray dataset.

    Notes
    -----
        - The output variable names will be suffixed with the provided suffix parameter.
        - If your given xr.Dataset was provided using a selection by values or indices, it is suggested to use the expand_and_assign_coords() function in order to contain the correct values of the dimensions and coordinates.


    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing the state and state variables.
    state_variables : list len(n)
        List of state variables to be used in the Kalman filter.
    observation_variables : list len(p)
        List of observed variables to be used in the Kalman filter.
    initial_state_estimation : np.ndarray shape(n)
        Initial State estimation at the first ``dim`` step (e.g. first timestep in the input DataSet).
    initial_covariance_matrix : np.ndarray shape(n,n)
        Initial Covarince Matrix corresponding to the initial estimation of the state variables.
    M : array-like, shape (T, n, n)
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
            - state_forecast<suffix>:
                - DataArray containing the estimated states over time.
                - Dimensions: time, state_name
            - covariance_forecast<suffix>:
                - DataArray containing the covariance of the estimated states over time.
                - Dimensions: time, state_name, state_name_copy
            - state_assimilation<suffix>:
                - DataArray containing the estimated states over time.
                - Dimensions: time, state_name
            - covariance_assimilation<suffix>:
                - DataArray containing the covariance of the estimated states over time.
                - Dimensions: time, state_name, state_name_copy
            - log_likelihod<suffix>:
                - DataArray containing the log-likelihood values for each timestep.
                - Dimensions: time
            # Not yet implemented.
            # - kalman_gain<suffix>: DataArray containing the Kalman gain values for each timestep.
            #     - Dimensions: time

    """
    # raise NotImplementedError("Function full deployed yet!")
    # function to create new names from a list of strings
    join_names = lambda l: "".join(l)

    n = len(state_variables)
    p = len(observation_variables)
    T = np.size(ds[dim])

    # check the dimensions:
    assert np.shape(initial_covariance_matrix) == (
        n,
        n,
    ), f"Mismatch in dimensions of initial covariance matrix : {np.shape(initial_covariance_matrix)} but should be {(n,n)}"
    assert np.shape(M) == (
        T,
        n,
        n,
    ), f"Mismatch in dimensions of R : {np.shape(M)} but should be {(T,n,n)}"
    assert np.shape(Q) == (
        T,
        n,
        n,
    ), f"Mismatch in dimensions of Q : {np.shape(Q)} but should be {(T,n,n)}"
    assert np.shape(H) == (
        p,
        n,
    ), f"Mismatch in dimensions of R : {np.shape(H)} but should be {(p,n)}"
    assert np.shape(R) == (
        p,
        p,
    ), f"Mismatch in dimensions of R : {np.shape(R)} but should be {(p,p)}"

    # create a list of numpy array to be used for the Kalman iteration
    observation_list = []
    state_list = []
    for var in observation_variables:
        observation_list.append(ds[var].values.flatten())
    # for var in state_variables:
    #     state_list.append(ds[var].isel({dim: estimation_idx}).values.flatten())

    unused, observations = input_arrays(
        observation_list=observation_list, state_list=state_list
    )

    # # initial_state_estimation has shape (1,n) but we need it to be (n)
    # initial_state_estimation = initial_state_estimation[0]

    # ---------------
    # run the Kalman_SEM algorithm
    # ---------------
    (x_f, P_f, x_a, P_a, log_likelihod, K_a) = Kalman_filter_time_dependent(
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
    Notes: !!!!!!!!!! NOT READY !!!!!!!!!!
    Run the Kalman smoother on the input dataset and return the results in an xarray Dataset.

    This function applies the Kalman smoother algorithm on the specified observations and state variables in the given xarray dataset.
    The results are stored in a new xarray dataset.

    Notes
    -----
        - The initial state estimation will use the ``state_variables`` provided in ``ds``
          and use the values of the array at position ``estimation_idx`` to create the initial state estimation used for the kalman smoother.
        - The output variable names will be suffixed with the provided suffix parameter.
        - If your given xr.Dataset was provided using a selection by values or indices, it is suggested to use the expand_and_assign_coords() function in order to contain the correct values of the dimensions and coordinates.

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
        Index used for the initial estimation of the state from along 'dim'.
        Default to 0.
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
            - state_forecast<suffix>:
                - DataArray containing the estimated states over time.
                - Dimensions: time, state_name
            - covariance_forecast<suffix>:
                - DataArray containing the covariance of the estimated states over time.
                - Dimensions: time, state_name, state_name_copy
            - state_assimilation<suffix>:
                - DataArray containing the estimated states over time.
                - Dimensions: time, state_name
            - covariance_assimilation<suffix>:
                - DataArray containing the covariance of the estimated states over time.
                - Dimensions: time, state_name, state_name_copy
            - state_smooth<suffix>:
                - DataArray containing the estimated states over time.
                - Dimensions: time, state_name
            - covariance_smooth<suffix>:
                - DataArray containing the covariance of the estimated states over time.
                - Dimensions: time, state_name, state_name_copy
            - log_likelihod<suffix>:
                - DataArray containing the log-likelihood values for each timestep.
                - Dimensions: time
            - kalman_gain<suffix>:
                - !!! Not yet implemented. !!!
                - DataArray containing the Kalman gain values for each timestep.
                - Dimensions: time
    """
    # raise NotImplementedError("Function full deployed yet!")
    # function to create new names from a list of strings
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
    ), f"Mismatch in dimensions of M : {np.shape(M)} but should be {(n,n)}"
    assert np.shape(Q) == (
        n,
        n,
    ), f"Mismatch in dimensions of Q : {np.shape(Q)} but should be {(n,n)}"
    assert np.shape(H) == (
        p,
        n,
    ), f"Mismatch in dimensions of H : {np.shape(H)} but should be {(p,n)}"
    assert np.shape(R) == (
        p,
        p,
    ), f"Mismatch in dimensions of R : {np.shape(R)} but should be {(p,p)}"

    # create a list of numpy array to be used for the Kalman iteration
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


def xarray_Kalman_smoother_time_dependent(
    ds: xr.Dataset,
    state_variables: Iterable[str],
    observation_variables: Iterable[str],
    initial_covariance_matrix: np.ndarray,
    M: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    dim="time",
    estimation_idx=-1,
    suffix: str = "",
) -> xr.Dataset:
    """
    !!!!!!!!!! NOT READY !!!!!!!!!! Run the Kalman smoother in the time
    dependent version on the input dataset and return the results in an xarray
    Dataset.

    This function applies the Kalman smoother algorithm on the specified observations and state variables in the given xarray dataset.
    The results are stored in a new xarray dataset.

    Notes
    -----
        - The initial state estimation will use the ``state_variables`` provided in ``ds``
        and use the values of the array at position ``estimation_idx`` to create the initial state estimation used for the kalman smoother.
        - The output variable names will be suffixed with the provided suffix parameter.
        - If your given xr.Dataset was provided using a selection by values or indices, it is suggested to use the expand_and_assign_coords() function in order to contain the correct values of the dimensions and coordinates.

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
        Index used for the initial estimation of the state from along 'dim'.
        Default to 0.
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
            - state_forecast<suffix>:
                - DataArray containing the estimated states over time.
                - Dimensions: time, state_name
            - covariance_forecast<suffix>:
                - DataArray containing the covariance of the estimated states over time.
                - Dimensions: time, state_name, state_name_copy
            - state_assimilation<suffix>:
                - DataArray containing the estimated states over time.
                - Dimensions: time, state_name
            - covariance_assimilation<suffix>:
                - DataArray containing the covariance of the estimated states over time.
                - Dimensions: time, state_name, state_name_copy
            - state_smooth<suffix>:
                - DataArray containing the estimated states over time.
                - Dimensions: time, state_name
            - covariance_smooth<suffix>:
                - DataArray containing the covariance of the estimated states over time.
                - Dimensions: time, state_name, state_name_copy
            - log_likelihod<suffix>:
                - DataArray containing the log-likelihood values for each timestep.
                - Dimensions: time
            - # kalman_gain<suffix>:
                - # DataArray containing the Kalman gain values for each timestep.
                - # Dimensions: time
    """
    # raise NotImplementedError("Function full deployed yet!")
    # function to create new names from a list of strings
    join_names = lambda l: "".join(l)

    n = len(state_variables)
    p = len(observation_variables)
    T = np.size(ds[dim])
    # check the dimensions:
    assert np.shape(initial_covariance_matrix) == (
        n,
        n,
    ), f"Mismatch in dimensions of initial covariance matrix : {np.shape(initial_covariance_matrix)} but should be {(n,n)}"
    assert np.shape(M) == (
        T,
        n,
        n,
    ), f"Mismatch in dimensions of M : {np.shape(M)} but should be {(T,n,n)}"
    assert np.shape(Q) == (
        T,
        n,
        n,
    ), f"Mismatch in dimensions of Q : {np.shape(Q)} but should be {(T,n,n)}"
    assert np.shape(H) == (
        p,
        n,
    ), f"Mismatch in dimensions of H : {np.shape(H)} but should be {(p,n)}"
    assert np.shape(R) == (
        p,
        p,
    ), f"Mismatch in dimensions of R : {np.shape(R)} but should be {(p,p)}"

    # create a list of numpy array to be used for the Kalman iteration
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
    (
        x_f,
        P_f,
        x_a,
        P_a,
        x_s,
        P_s,
        log_likelihod,
        P_s_lag,
    ) = Kalman_smoother_time_dependent(
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

    Notes
    -----
        - The state_varianbles can contain latent variables which are not found in the observation_variables
        - The output variable names will be suffixed with the provided suffix parameter.
        - If your given xr.Dataset was provided using a selection by values or indices, it is suggested to use the expand_and_assign_coords() function in order to contain the correct values of the dimensions and coordinates.

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

        Output Dimensions:
            - time: The time dimension of the input dataset.
            - state_name: Dimension representing the state variables.
            - state_name_copy: Dimension representing a copy of the state variables.

        Output Coordinates:
            - time: Coordinates corresponding to the time of the input dataset.
            - state_name: Coordinates corresponding to the state variables.
            - state_name_copy: Coordinates corresponding to the copy of the state variables.
            - kalman_iteration: Coordinates representing the Kalman SEM iteration index.

        Output Variables:
            - states<suffix>:
                - DataArray containing the estimated states over time.
                - Dimensions: time, state_name
            - covariance<suffix>:
                - DataArray containing the covariance of the estimated states over time.
                - Dimensions: time, state_name, state_name_copy
            - M<suffix>:
                - DataArray containing the transition matrix M.
                - Dimensions: state_name, state_name_copy
            - Q<suffix>:
                - DataArray containing the observation noise covariance matrix Q.
                - Dimensions: state_name, state_name_copy
            - log_likelihod<suffix>:
                - DataArray containing the log-likelihood values for each Kalman SEM iteration.
                - Dimensions: kalman_iteration
    """

    # function to create new names from a list of strings
    join_names = lambda l: "".join(l)

    # make sure that the variables in observation_variables are in same order as in state_variables
    state_variables = ordered_like(state_variables, observation_variables)
    # check that state_variables is a subset sorted like observation_variables
    assert_ordered_subset(observation_variables, state_variables)

    # create a list of numpy array to be used for the Kalman iteration
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
            kalman_iteration=np.arange(nb_iter_SEM),
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
        dims=["kalman_iteration"],
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

    Notes
    -----
        - The state_varianbles can contain latent variables which are not found in the observation_variables
        - The output variable names will be suffixed with the provided suffix parameter.
        - If your given xr.Dataset was provided using a selection by values or indices, it is suggested to use the expand_and_assign_coords() function in order to contain the correct values of the dimensions and coordinates.

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

        Output Dimensions:
            - time: The time dimension of the input dataset. shape(t)
            - state_name: Dimension representing the state variables. shape(n)
            - state_name_copy: Dimension representing a copy of the state variables. shape(n)

        Output Coordinates:
            - time: Coordinates corresponding to the time of the input dataset.
            - state_name: Coordinates corresponding to the state variables.
            - state_name_copy: Coordinates corresponding to the copy of the state variables.
            - kalman_iteration: Coordinates representing the Kalman SEM iteration index.

        Output Variables:
            - states<suffix>:
                - DataArray containing the estimated states over time.
                - Dimensions: time, state_name
            - covariance<suffix>:
                - DataArray containing the covariance of the estimated states over time.
                - Dimensions: time, state_name, state_name_copy
            - M<suffix>:
                - DataArray containing the transition matrix M.
                - Dimensions: state_name, state_name_copy
            - Q<suffix>:
                - DataArray containing the observation noise covariance matrix Q.
                - Dimensions: state_name, state_name_copy
            - log_likelihod<suffix>:
                - DataArray containing the log-likelihood values for each Kalman SEM iteration.
                - Dimensions: kalman_iteration
    """

    # function to create new names from a list of strings
    join_names = lambda l: "".join(l)

    # make sure that the variables in observation_variables are in same order as in state_variables
    state_variables = ordered_like(state_variables, observation_variables)
    # check that state_variables is a subset sorted like observation_variables
    assert_ordered_subset(observation_variables, state_variables)

    # create a list of numpy array to be used for the Kalman iteration
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
            kalman_iteration=np.arange(nb_iter_SEM),
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
        dims=["kalman_iteration"],
    )

    return result


def xarray_Kalman_SEM_full_output(
    ds: xr.Dataset,
    observation_variables: Iterable[str],
    state_variables: Iterable[str],
    nb_iter_SEM: int = 30,
    variance_obs_comp: float = 0.0001,
    suffix: str = "",
) -> xr.Dataset:
    """
    Run the Kalman SEM algorithm on the input dataset and return the results in
    an xarray Dataset. The output will contain the values of the returns for each iteration.
    The function uses ``kalman.Kalman_SEM_full_output``.

    This function applies the Kalman_SEM algorithm on the specified observations and state variables in the given xarray dataset.
    It performs a specified number of iterations and computes the state estimates, covariance, transition matrix, and
    process noise covariance matrix. The results are stored in a new xarray dataset.

    Notes
    -----
        - The state_varianbles can contain latent variables which are not found in the observation_variables
        - The output variable names will be suffixed with the provided suffix parameter.
        - If your given xr.Dataset was provided using a selection by values or indices, it is suggested to use the expand_and_assign_coords() function in order to contain the correct values of the dimensions and coordinates.

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
            - kalman_iteration: Coordinates representing the Kalman SEM iteration index.

        Output Variables:
            - states<suffix>:
                -DataArray containing the estimated states over time.
                - Dimensions: time, state_name
            - covariance<suffix>:
                - DataArray containing the covariance of the estimated states over time.
                - Dimensions: time, state_name, state_name_copy
            - M<suffix>:
                - DataArray containing the transition matrix M.
                - Dimensions: state_name, state_name_copy
            - Q<suffix>:
                - DataArray containing the observation noise covariance matrix Q.
                - Dimensions: state_name, state_name_copy
            - log_likelihod<suffix>:
                - DataArray containing the log-likelihood values for each Kalman SEM iteration.
                - Dimensions: kalman_iteration
    """

    # function to create new names from a list of strings
    join_names = lambda l: "".join(l)

    # make sure that the variables in observation_variables are in same order as in state_variables
    state_variables = ordered_like(state_variables, observation_variables)
    # check that state_variables is a subset sorted like observation_variables
    assert_ordered_subset(observation_variables, state_variables)

    # create a list of numpy array to be used for the Kalman iteration
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
    ) = Kalman_SEM_full_output(
        x=states, y=observations, H=H, R=R, nb_iter_SEM=nb_iter_SEM
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
            kalman_iteration=np.arange(nb_iter_SEM),
        )
    )
    # store x_s
    new_var = join_names(["states", suffix])
    result[new_var] = xr.DataArray(
        data=x_s,
        dims=["kalman_iteration", "time", "state_name"],
    )
    # store P_s
    new_var = join_names(["covariance", suffix])
    result[new_var] = xr.DataArray(
        data=P_s,
        dims=["kalman_iteration", "time", "state_name", "state_name_copy"],
    )
    # store M
    new_var = join_names(["M", suffix])
    result[new_var] = xr.DataArray(
        data=M,
        dims=["kalman_iteration", "state_name", "state_name_copy"],
    )
    # store Q
    new_var = join_names(["Q", suffix])
    result[new_var] = xr.DataArray(
        data=Q,
        dims=["kalman_iteration", "state_name", "state_name_copy"],
    )
    # store the log_likelihod
    new_var = join_names(["log_likelihod", suffix])
    result[new_var] = xr.DataArray(
        data=log_likelihod,
        dims=["kalman_iteration"],
    )

    # shift the kalman_iteration to the end
    # result = result.transpose(dims = ["time", "state_name", "state_name_copy", "kalman_iteration"])

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
        select_dict (dict): Dictionary of selection or index selection containing names and values of the coords and dims to expand.

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
            raise NotImplementedError("This seems not to be implemented!")
    ds1_expanded = ds1_expanded.expand_dims(selection_vars)

    return ds1_expanded


def all_choords_as_dim(
    ds: Union[xr.Dataset, xr.DataArray]
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Expand all coordinates in the xarray Dataset or DataArray as new
    dimensions.

    This function expands each coordinate in the xarray Dataset or DataArray as a new dimension.
    If a coordinate cannot be expanded (e.g., it already exists as a dimension), it is skipped.

    Parameters:
        - ds (Union[xr.Dataset, xr.DataArray]): The input xarray Dataset or DataArray.

    Returns:
        - Union[xr.Dataset, xr.DataArray]: The xarray object with all coordinates expanded as dimensions.

    Examples:
        >>> # Example 1: Expand coordinates in a Dataset
        >>> import xarray as xr
        >>> data = [[1, 2], [3, 4]]
        >>> coords = {'time': [0, 1], 'latitude': [10, 20], 'longitude': [30, 40]}
        >>> ds = xr.Dataset({'data': (['time', 'latitude'], data)}, coords=coords)
        >>> expanded_ds = all_choords_as_dim(ds)
        >>> print(expanded_ds)

        >>> # Example 2: Expand coordinates in a DataArray
        >>> import xarray as xr
        >>> data = [1, 2, 3, 4]
        >>> coords = {'latitude': [10, 20], 'longitude': [30, 40]}
        >>> da = xr.DataArray(data, coords=coords, dims=['location'])
        >>> expanded_da = all_choords_as_dim(da)
        >>> print(expanded_da)
    """
    for dim in ds.coords:
        try:
            ds = ds.expand_dims(dim)
        except:
            pass
    return ds


def all_dims_as_choords(
    ds: Union[xr.Dataset, xr.DataArray]
) -> Union[xr.Dataset, xr.DataArray]:
    """
    TODO: UPDATE NEEDED!
    Expand all coordinates in the xarray Dataset or DataArray as new
    dimensions.

    This function expands each coordinate in the xarray Dataset or DataArray as a new dimension.
    If a coordinate cannot be expanded (e.g., it already exists as a dimension), it is skipped.

    Parameters:
        - ds (Union[xr.Dataset, xr.DataArray]): The input xarray Dataset or DataArray.

    Returns:
        - Union[xr.Dataset, xr.DataArray]: The xarray object with all coordinates expanded as dimensions.
    """
    for dim in ds.dims:
        try:
            ds = ds.assign_coords({dim: ds[dim]})
        except:
            pass
    return ds


# Functions to add one or more variables to a DataSet


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

    Examples:
        >>> ds1 = xr.Dataset(
        ... {"var1": (("x", "y"), np.zeros((3, 3)))},
        ... coords={"x": [1, 2, 3], "y": [4, 5, 6]}
        ... )
        >>> ds2 = xr.Dataset(
        ... {"var2": (("x", "y"), np.ones((3, 3)))},
        ... coords={"x": [2, 3, 4], "y": [5, 6, 7]}
        ... )
        >>> new_dimension = "run"
        >>> add_all_empty_dataarray(ds1, ds2, new_dimension)
        >>> print(ds1)
        ... <xarray.Dataset>
        ... Dimensions:  (x: 3, y: 3, run: 1)
        ... Coordinates:
        ...   * x        (x) int64 1 2 3
        ...   * y        (y) int64 4 5 6
        ...   * run      (run) int64 0
        ... Data variables:
        ...     var1     (x, y) float64 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
        ...     var2     (run, x, y) float64 nan nan nan nan nan nan nan nan nan
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
    _force: bool = False,
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

    Examples:
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

    if _force:
        # Iterate over all variables in ds2
        for var in ds2.data_vars:
            # Assign values from ds2 to corresponding variables in ds1 at the selection coordinates
            ds1[var].loc[select_dict1] = ds2[var].loc[select_dict2].to_numpy()
    else:
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

    # for the first iteration use the result to create the new coordinates and variables needed in the output DataSet
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


def run_function_on_multiple_subdatasets(
    processing_function: Callable[[xr.Dataset], xr.Dataset],
    parent_dataset: xr.Dataset,
    subdataset_selections: Iterable[Dict],
    func_args: Dict = {},
    func_kwargs: Dict = {},
) -> xr.Dataset:
    """
    Apply a processing function to multiple subdatasets and merge the results.

    This function applies the provided processing function to multiple subdatasets from the parent dataset.
    It uses the subdataset_selections iterable to select specific subsets of data from the parent dataset.
    The processing function should take an xr.Dataset as its input and return an xr.Dataset as its output.

    Notes:
        - the resulting arrays contain np.nan wherever the

    Parameters:
        processing_function (Callable): The processing function to be applied to each subdataset.
            The function should take an xr.Dataset as its input and return an xr.Dataset as its output.
        parent_dataset (xr.Dataset): The parent dataset containing the subdatasets to process.
        subdataset_selections (Iterable[Dict]): The iterable of selection dictionaries for each subdataset.
            Each dictionary specifies the subset of data to select from the parent dataset.
        func_args (Dict, optional): Additional positional arguments to be passed to the processing function.
            Default is an empty dictionary.
        func_kwargs (Dict, optional): Additional keyword arguments to be passed to the processing function.
            The 'ds' argument is automatically updated with the selected subdataset.
            Default is an empty dictionary.

    Returns:
        xr.Dataset: The merged dataset containing the results of applying the processing function to each subdataset.

    Example:
        >>> def multiply_by_scalar(ds: xr.Dataset, scalar: float) -> xr.Dataset:
        ...     return ds * scalar
        >>> ds = xr.Dataset(
        ...     {"data": (("x", "y", "time"), [
        ...         [[1, 2, 3, 4], [6, 7, 8, 9], [11, 12, 13, 14]],
        ...         [[16, 17, 18, 19], [21, 22, 23, 24], [26, 27, 28, 29]],
        ...     ])},
        ...     coords={"time": range(4), "x": [2, 3], "y": [5, 6, 7]},
        ... )
        >>> subdataset_selections = [{"x": 2, "y": 5}, {"x": 3, "y": 6}]
        >>> func_kwargs = {"scalar": 2}
        >>> processing_function = multiply_by_scalar
        >>> result = run_function_on_multiple_subdatasets(
        ...     processing_function=processing_function,
        ...     parent_dataset=ds,
        ...     subdataset_selections=subdataset_selections,
        ...     func_kwargs=func_kwargs,
        ... )
        >>> print(result)
        <xarray.Dataset>
        Dimensions:  (time: 4, x: 2, y: 2)
        Coordinates:
        * time     (time) int32 0 1 2 3
        * x        (x) int32 2 3
        * y        (y) int32 5 6
        Data variables:
            data     (x, y, time) float64 2.0 4.0 6.0 8.0 nan ... 42.0 44.0 46.0 48.0
    """
    result = []
    for subdataset_selection in subdataset_selections:
        subdataset = parent_dataset.sel(subdataset_selection, drop=True)
        func_kwargs.update(ds=subdataset)
        try:
            res = processing_function(*func_args, **func_kwargs)
            if not isinstance(res, xr.Dataset):
                raise ValueError(
                    f"The processing function should return an xr.Dataset, but it returned {type(res)}"
                )
            else:
                res = expand_and_assign_coords(
                    res, parent_dataset, select_dict=subdataset_selection
                )
                result.append(res)

        except Exception as ex:
            # using traceback https://stackoverflow.com/a/47659065/16372843
            traceback_message = traceback.format_exc()
            message = f"{type(ex).__name__} occurred during processing. Traceback is:\n{traceback_message}"
            # TODO : use warning instead of print
            print(message)

    return xr.merge(result)


def run_function_on_multiple_time_slices(
    processing_function: Callable[[xr.Dataset], xr.Dataset],
    parent_dataset: xr.Dataset,
    time_slices: Iterable[Dict],
    func_args: Dict = {},
    func_kwargs: Dict = {},
    new_dimension: str = "start_time",
) -> xr.Dataset:
    """
    Apply a processing function to multiple subdatasets and merge the results.

    This function applies the provided processing function to multiple subdatasets from the parent dataset.
    It uses the time_slices iterable to select specific subsets of data from the parent dataset.
    The processing function should take an xr.Dataset as its input and return an xr.Dataset as its output.

    Notes:
        - the resulting arrays contain np.nan wherever the

    Parameters:
        processing_function (Callable): The processing function to be applied to each subdataset.
            The function should take an xr.Dataset as its input and return an xr.Dataset as its output.
        parent_dataset (xr.Dataset): The parent dataset containing the subdatasets to process.
        time_slices (Iterable[Dict]): The iterable of selection dictionaries for each subdataset.
            Each dictionary specifies the subset of data to select from the parent dataset.
        func_args (Dict, optional): Additional positional arguments to be passed to the processing function.
            Default is an empty dictionary.
        func_kwargs (Dict, optional): Additional keyword arguments to be passed to the processing function.
            The 'ds' argument is automatically updated with the selected subdataset.
            Default is an empty dictionary.

    Returns:
        xr.Dataset: The merged dataset containing the results of applying the processing function to each subdataset.

    Example:
        >>> def multiply_by_scalar(ds: xr.Dataset, scalar: float) -> xr.Dataset:
        ...     return ds * scalar
        >>> ds = xr.Dataset(
        ...     {"data": (("x", "y", "time"), [
        ...         [[1, 2, 3, 4], [6, 7, 8, 9], [11, 12, 13, 14]],
        ...         [[16, 17, 18, 19], [21, 22, 23, 24], [26, 27, 28, 29]],
        ...     ])},
        ...     coords={"time": range(4), "x": [2, 3], "y": [5, 6, 7]},
        ... )
        >>> subdataset_selections = [{"x": 2, "y": 5}, {"x": 3, "y": 6}]
        >>> func_kwargs = {"scalar": 2}
        >>> processing_function = multiply_by_scalar
        >>> result = run_function_on_multiple_subdatasets(
        ...     processing_function=processing_function,
        ...     parent_dataset=ds,
        ...     subdataset_selections=subdataset_selections,
        ...     func_kwargs=func_kwargs,
        ... )
        >>> print(result)
        <xarray.Dataset>
        Dimensions:  (time: 4, x: 2, y: 2)
        Coordinates:
        * time     (time) int32 0 1 2 3
        * x        (x) int32 2 3
        * y        (y) int32 5 6
        Data variables:
            data     (x, y, time) float64 2.0 4.0 6.0 8.0 nan ... 42.0 44.0 46.0 48.0
    """
    result = []
    expanddataset = xr.Dataset(
        coords={f"{new_dimension}": [s["time"].start for s in time_slices]}
    )

    for time_slice_selection in time_slices:
        expand_selection = {f"{new_dimension}": time_slice_selection["time"].start}

        subdataset = parent_dataset.sel(time_slice_selection, drop=True)
        func_kwargs.update(ds=subdataset)
        try:
            res = processing_function(*func_args, **func_kwargs)
            if not isinstance(res, xr.Dataset):
                raise ValueError(
                    f"The processing function should return an xr.Dataset, but it returned {type(res)}"
                )
            else:
                res = expand_and_assign_coords(
                    res, expanddataset, select_dict=expand_selection
                )
                result.append(res)

        except Exception as e:
            warn(f"ValueError: Error occurred during processing: {type(e)}")

    return xr.merge(result)


# Analysis functions


def forcast_from_kalman(
    ds_kalman_SEM: xr.Dataset,
    ds_state_covariance: xr.Dataset,
    state_var_name: str = "states",
    covariance_var_name: str = "covariance",
    new_dimension: str = "horizon",
    forecast_dim: str = "time",
    forecast_length: int = 30,
) -> xr.Dataset:
    """
    Generate forecasts from the Kalman SEM results for a given forecast length.

    This function generates forecasts based on the Kalman SEM results and a provided dataset containing the state and covariance dataset.
    It creates a new dataset with the forecasted states and covariance matrices for a specified forecast length.



    Parameters:
        ds_kalman_SEM (xr.Dataset): The Kalman SEM results dataset.
            Has to contain the variables "M", "Q", "states", "covariance".
        ds_state_covariance (xr.Dataset): The state covariance dataset.
        state_var_name (str): The variable name for the states in ds_state_covariance. Default is "states".
        covariance_var_name (str): The variable name for the covariance matrices in ds_state_covariance. Default is "covariance".
        new_dimension (str): The name of the new dimension representing the forecast horizon. Default is "horizon".
        forecast_dim (str): The name of the dimension representing the time in ds_kalman_SEM. Default is "time".
        forecast_length (int): The length of the forecast horizon. Default is 30.

    Returns:
        xr.Dataset: The dataset containing the forecasted states and covariance matrices.

    Example:
        Generate forecasts with a forecast length of 30

        >>> # Create example data
        >>> time = np.array([0.0, 0.001, 0.002])
        >>> state_name = np.array(['x', 'y', 'z'])
        >>> state_name_copy = np.array(['x', 'y', 'z'])
        >>> states_data = np.array([
                [-1.798, 17.91, 5.0],
                [-1.785, 17.889, 5.1],
                [-1.792, 17.85, 5.2]
                ])
        >>> covariance_data = np.array([
                [[0.000105, 0.000086, 0.0012],
                [0.000086, 0.000131, -0.0015],
                [0.0012, -0.0015, 0.256]],
                #
                [[0.000075, 0.000021, 0.000184],
                [0.000021, 0.000077, -0.000201],
                [0.000184, -0.000201, 1.446]],
                #
                [[0.000078, 0.000018, -0.000034],
                [0.000018, 0.000079, 0.000024],
                [-0.000034, 0.000024, 1.553]]
                ])
        >>> M_data = np.array([
                [1.0, -0.000043, -0.0047],
                [0.000492, 1.0, 0.0067],
                [0.0056, -0.0012, 0.983]
                ])
        >>> Q_data = np.array([
                [0.003, 0.0029, -0.0077],
                [0.0029, 0.0032, 0.0102],
                [-0.0077, 0.0102, 3.028]
                ])

        >>> # Create the xarray Dataset
        >>> ds = xr.Dataset(coords={'time': time, 'state_name': state_name, 'state_name_copy': state_name_copy})
        >>> # Add data variables
        >>> ds['states'] = (('time', 'state_name'), states_data)
        >>> ds['covariance'] = (('time', 'state_name', 'state_name_copy'), covariance_data)
        >>> ds['M'] = (('state_name', 'state_name_copy'), M_data)
        >>> ds['Q'] = (('state_name', 'state_name_copy'), Q_data)

        >>> forecast = forcast_from_kalman(
                ds_kalman_SEM = ds,
                ds_state_covarinace = ds,
                state_var_name = "states",
                covariance_var_name = "covariance",
                forecast_length = 10,
        )

        >>> # Access the forecasted states and covariance matrices
        >>> forecast_states = forecast["states"]
        >>> forecast_covariance = forecast["covariance"]
        >>> # Access a specific forecasted state and covariance matrix at horizon index 5
        >>> state_at_horizon_5 = forecast_states.sel(horizon=5)
        >>> covariance_at_horizon_5 = forecast_covariance.sel(horizon=5)

    Notes:
        - You can use the ``from_standard_dataset`` function too.
    """
    # first create the corresponding result Dataset
    # copy coords from kalman_SEM results
    result = xr.Dataset(coords=ds_kalman_SEM.coords)
    # add new coordinate
    result = result.assign_coords({new_dimension: np.arange(forecast_length)})
    # assign empty data arrays to the result
    add_empty_dataarrays(
        result, ds_kalman_SEM[["states", "covariance"]], new_dimension=new_dimension
    )

    # make sure it is in the right order
    result = result.transpose(new_dimension, forecast_dim, ...)
    ds_kalman_SEM = ds_kalman_SEM.transpose(forecast_dim, ...)

    # assign the initial M and Q Matrices
    # this also drops the newdimension for M and Q
    result["M"] = ds_kalman_SEM["M"]
    result["Q"] = ds_kalman_SEM["Q"]

    # assign the initial state
    assign_variable_by_double_selection(
        ds1=result,
        da2=ds_state_covariance[state_var_name],
        var_name="states",
        select_dict1={new_dimension: 0},
        select_dict2={},
    )

    # assign the initial covarinace
    assign_variable_by_double_selection(
        ds1=result,
        da2=ds_state_covariance[covariance_var_name],
        var_name="covariance",
        select_dict1={new_dimension: 0},
        select_dict2={},
    )

    # It is importatnt to make sure that the order of the dimensions is
    # `forecast_dim`, state_name, state_name_copy, ...
    result = result.transpose(forecast_dim, "state_name", "state_name_copy", ...)
    # For the whole forecast length, compute the new state at each forecast step and the corresponding other stuff
    for idx in range(0, forecast_length - 1):
        # TODO: This for loop might also be negelectable with the einstein convention by using it as another indice.
        state_forecast, covariance_forecast = kalman_single_forecast(
            S=result.states.isel({new_dimension: idx}).values,
            C=result.covariance.sel({new_dimension: idx}).values,
            M=result.M.values,
            Q=result.Q.values,
        )
        # state forecast
        result["states"].loc[{new_dimension: idx + 1}] = state_forecast
        # covariance forecast
        result["covariance"].loc[{new_dimension: idx + 1}] = covariance_forecast

    return result


def to_standard_dataset(
    ds: xr.Dataset,
    states_variables: Union[str, Iterable[str]] = "all",
    new_dim: str = "states",
) -> xr.Dataset:
    """
    Convert a dataset to a standard dataset format for state variables.

    This function converts the provided dataset `ds` into a standard dataset format.
    This format includes the new dimension ``new_dim``.
    It creates a new xr.Dataset where each state variable becomes a separate coordinate in the "states" variable.

    Parameters:
        ds (xr.Dataset): The dataset to be converted.
        states_variables (str or Iterable[str]): The state variables to include in the conversion.
            - If "all", all data variables in `ds` will be considered as state variables.
            - If a string or iterable of strings is provided, only the specified variables will be considered as state variables.
        new_dim (str) : Name of the new dimension storing the values of states_variables. Default to "states".

    Returns:
        xr.Dataset: The converted dataset in the standard format.

            Output Variables:
                - "states" (dims: ["state_name", *ds.dims]): The state variables in the standard format.

            Output Dimensions:
                - "state_name": The dimension representing the state variables.
                - Additional dimensions inherited from the input dataset `ds`.

            Output Coordinates:
                - "state_name": The values of the state variable names.
                - Coordinates inherited from the input dataset `ds`.

    Example:
        >>> ds = xr.Dataset(
        ...     {"temperature": (("time", "latitude", "longitude"), temperature_data)},
        ...     coords={
        ...         "time": pd.date_range("2022-01-01", periods=365),
        ...         "latitude": [30, 40, 50],
        ...         "longitude": [-120, -110, -100],
        ...     },
        ... )
        >>> converted_ds = to_standard_dataset(ds, states_variables=["temperature"])
        >>> print(converted_ds)
        ... <xarray.Dataset>
        ... Dimensions:    (state_name: 1, time: 365, latitude: 3, longitude: 3)
        ... Coordinates:
        ...   * time       (time) datetime64[ns] 2022-01-01 ... 2022-12-31
        ...   * latitude   (latitude) int64 30 40 50
        ...   * longitude  (longitude) int64 -120 -110 -100
        ...   * state_name (state_name) <U11 'temperature'
        ... Data variables:
        ...     states     (state_name, time, latitude, longitude) float64 ...
    """

    if states_variables == "all":
        states_variables = list(ds.data_vars)

    result = xr.Dataset(coords=ds.coords)
    result = result.assign_coords(
        dict(
            state_name=states_variables,
            # state_name_copy = states_variables,
        )
    )
    # initlize the dataarray for the states using dimensions of the first state
    state = states_variables[0]
    # this should only use dimensional coordinates, thus only use these list of coords:
    coords = [result.state_name] + [ds[state].coords[var] for var in ds.dims]
    result[new_dim] = xr.DataArray(coords=coords)
    # fill it
    for state in states_variables:
        assign_variable_by_double_selection(
            ds1=result,
            da2=ds[state],
            var_name=new_dim,
            select_dict1=(dict(state_name=state)),
            select_dict2=dict(),
        )
    return result


def from_standard_dataset(
    ds: xr.Dataset,
    var_name: str = "states",
    dim_name: str = "state_name",
    suffix: str = "",
    prefix: str = "",
) -> xr.Dataset:
    """
    Convert the DataArray ``var_name`` from a standard dataset ``ds`` to a new Dataset.
    The input ``var_name`` need to have dimension ``dim_name``. The values of ``dim_name`` will be used to

    This function converts the provided dataset `ds` in the standard format back into the original dataset format.
    It merges the state variables stored in the ``var_name`` variable back into separate data variables.

    Notes:
        - The provided DataArray corresponding to ``ds`` [ ``var_name`` ] needs to have dimension ``dim_name`` as a valid dimension or coordinate.
        - The output names can be modified using the ``prefix`` and ``suffix`` args

    Parameters:
        ds (xr.Dataset): The dataset in the standard format.
        var_name (str): Variable name for which the separate data variables shall be used.
        dim_name (str): dimension and coordinate name use in the standard dataset to specify the dimension of "states" variables.
        suffix (str, optional) : Suffix to be appended to the variable names in the output dataset.
            - Default is an empty string.
        prefix (str, optional) : Prefix to be appended to the variable names in the output dataset.
            - Default is an empty string.

    Returns:
        xr.Dataset:
            The converted dataset in the original format.

    Examples:

        Create the dataset

        >>> ds_init = xr.Dataset(
        ...     {
        ...     "temperature": (("time", "latitude", "longitude"), temperature_data),
        ...     "pressure": (("time", "latitude", "longitude"), pressure_data)
        ...     },
        ...     coords={
        ...         "time": pd.date_range("2022-01-01", periods=365),
        ...         "latitude": [30, 40, 50],
        ...         "longitude": [-120, -110, -100],
        ...     },
        ... )
        >>> states_data = n.concatenate((temperature_data, pressure_data), axis=0)
        >>> ds = xr.Dataset(
        ...     {"weather_states": (("state_name", *ds.dims), states_data)},
        ...     coords={
        ...         "state_name": ["temperature", "pressure"],
        ...         "time": pd.date_range("2022-01-01", periods=365),
        ...         "latitude": [30, 40, 50],
        ...         "longitude": [-120, -110, -100],
        ...     },
        ... )

        Convert the dataset

        >>> converted_ds = from_standard_dataset(ds, var_name = "weather_states, dim_name = "state_name)
        >>> print(converted_ds)
        ... <xarray.Dataset>
        ... Dimensions:    (time: 365, latitude: 3, longitude: 3)
        ... Coordinates:
        ...   * time       (time) datetime64[ns] 2022-01-01 ... 2022-12-31
        ...   * latitude   (latitude) int64 30 40 50
        ...   * longitude  (longitude) int64 -120 -110 -100
        ... Data variables:
        ...     temperature (time, latitude, longitude) float64 ...
        ...     pressure    (time, latitude, longitude) float64 ...

    """
    join_names = lambda l: "".join(l)

    variables = ds.coords[dim_name].values
    data_vars = {
        join_names(
            [prefix, var, suffix]
        ): ds[  # var names is modified by prefix and suffix
            var_name
        ].sel(
            {f"{dim_name}": var}
        )  # values
        for var in variables
    }
    result = xr.Dataset(data_vars, coords=ds.coords)
    # now drop the ``dim_name``
    return result.drop_vars(dim_name)


def perfect_forcast(
    ds: xr.Dataset,
    states_var_name: str = "states",
    new_dimension: str = "horizon",
    forecast_dim: str = "time",
    forecast_length: int = 30,
) -> xr.Dataset:
    """
    Generate perfect forecasts for a given dataset for a specified forecast
    length along the provided forecast dimension. Results will be stored in a
    new xr.Dataset along the given new dimension.

    This function generates perfect forecasts for a provided dataset containing a DataArray containing the states.

    Notes:
    The function expects a dataset of the same form as the `to_standard_dataset` function returns:
    - States array with dimensions [``forecast_dim``, "state_name"].
    Note that the forecast_length includes the current timestep.
    - ``forecast_length`` = 1 returns the same dataset with additional coordinate ``new_dimesion`` and value for it.

    Parameters:
        ds (xr.Dataset): The dataset containing the states.
            Must contain the variable specified by `states_var_name`.
        states_var_name (str): The variable name for the states in `ds`. Default is "states".
        new_dimension (str): The name of the new dimension representing the forecast horizon. Default is "horizon".
        forecast_dim (str): The name of the dimension representing the forecast dimension (e.g. "time") in `ds`. Default is "time".
        forecast_length (int): The length of the forecast horizon. Default is 30.

    Returns:
        xr.Dataset: The dataset containing the forecasted states and covariance matrices.

        Variables:
            - "states" (dims: ["horizon", "state_name"]): The forecasted states at each horizon step.

        Dimensions:
            - ``new_dimension``: The forecast horizon dimension.
            - Additional dimensions inherited from the input dataset `ds`.

        Coordinates:
            - ``new_dimension``: The values of the forecast horizon dimension, ranging from 0 to `forecast_length - 1`.
            - Coordinates inherited from the input dataset `ds`.

    Examples
    --------
    >>> # Create a dataset with states
    >>> ds = xr.Dataset(
    ... {
    ...     "states_model": (("time", "state_name"), np.arange(30).reshape(10, 3)),
    ...     "other_var": (("time",), np.arange(10)),
    ... },
    ... coords={
    ...     "time": np.arange(10),
    ...     "state_name": ["state1", "state2", "state3"],
    ... },
    ... )

    >>> # Generate perfect forecasts
    >>> result = perfect_forecast(ds, states_var_name="states_model", forecast_length=5, forecast_dim="time)
    >>> print(result)
    ... <xarray.Dataset>
    ... Dimensions:       (time: 10, state_name: 3, horizon: 5)
    ... Coordinates:
    ... * time          (time) int32 0 1 2 3 4 5 6 7 8 9
    ... * state_name    (state_name) <U6 'state1' 'state2' 'state3'
    ... * horizon       (horizon) int32 0 1 2 3 4
    ... Data variables:
    ...     states_model  (horizon, time, state_name) float64 0.0 1.0 2.0 ... nan nan
    ...     other_var     (horizon, time) float64 nan nan nan nan ... nan nan nan nan
    """

    # first create the corresponding result Dataset
    # copy coords from kalman_SEM results
    result = xr.Dataset(coords=ds.coords)
    # add new coordinate
    result = result.assign_coords({new_dimension: np.arange(forecast_length)})
    # assign empty data arrays to the result
    add_empty_dataarrays(result, ds, new_dimension=new_dimension)

    # make sure it is in the right order
    result = result.transpose(new_dimension, forecast_dim, ...)
    ds = ds.transpose(forecast_dim, ...)

    # assign the initial state
    assign_variable_by_double_selection(
        ds1=result,
        da2=ds[states_var_name],
        var_name=states_var_name,
        select_dict1={new_dimension: 0},
        select_dict2={},
    )

    maximum_forecast_index = len(ds[forecast_dim])
    # For the whole forecast length, compute the new state at each forecast step and the corresponding other stuff
    for idx in range(1, forecast_length):
        # state forecast
        # create insert range. it will get less and less with the horizon length.:
        insert_range = result[forecast_dim].isel(
            {forecast_dim: slice(0, maximum_forecast_index - idx)}
        )
        insert_select_dict = {
            new_dimension: idx,
            forecast_dim: insert_range,
        }
        # takes from old values of horizon == 0
        origin_forcast_range = result[forecast_dim].isel(
            {forecast_dim: slice(idx, maximum_forecast_index)}
        )
        origin_select_dict = {
            new_dimension: 0,
            forecast_dim: origin_forcast_range,
        }
        # state forecast
        result[states_var_name].loc[insert_select_dict] = (
            result[states_var_name].loc[origin_select_dict].values
        )

    return result
