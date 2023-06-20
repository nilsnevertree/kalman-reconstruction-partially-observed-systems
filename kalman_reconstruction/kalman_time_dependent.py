"""
Module containing time dependent version of the Kalman filter, smoother and
stochastic estimation maximization alogorithm.

The module includes
"""

from typing import Tuple

import numpy as np
import scipy as sp

from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from kalman_reconstruction.statistics import gaussian_kernel_1D


def Kalman_filter_time_dependent(
    y: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
    M: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the linear and Gaussian Kalman filter to estimate the state of a
    time-dependent system.

    This function applies the Kalman filter to a time-dependent system with linear dynamics and Gaussian noise.

    The underlying Kalman equations are:
    x(t) = M x(t+dt) + Q
    y(t) = H x(t) + R

    Parameters
    ----------
    y : array-like, shape (T, p)
        Observations of the system state observed over time.
    x0 : array-like, shape (n,)
        Initial estimate of the system state.
    P0 : array-like, shape (n, n)
        Initial estimate of the error covariance matrix of the system state.
    M : array-like, shape (T, n, n)
        Time-varying state transition matrix.
    Q : array-like, shape (T, n, n)
        Time-varying process noise covariance matrix.
    H : array-like, shape (p, n)
        Observation matrix that maps the state space to the observation space.
    R : array-like, shape (p, p)
        Covariance matrix representing the observation noise.

    Returns
    -------
    x_f : array-like, shape (T, n)
        Forecasted state estimates.
    P_f : array-like, shape (T, n, n)
        Forecasted error covariance matrices of the state estimates.
    x_a : array-like, shape (T, n)
        Updated state estimates after assimilating the observations.
    P_a : array-like, shape (T, n, n)
        Updated error covariance matrices of the state estimates after assimilating the observations.
    loglik : array-like, shape (T,)
        Log-likelihood values of the observations.
    K_a : array-like, shape (T, n, p)
        Kalman gain matrices used in the analysis step.
    """

    # shapes
    n = len(x0)
    T, p = np.shape(y)

    # Kalman initialization
    x_f = np.zeros((T, n))  # forecast state
    P_f = np.zeros((T, n, n))  # forecast error covariance matrix
    x_a = np.zeros((T, n))  # analysed state
    P_a = np.zeros((T, n, n))  # analysed error covariance matrix
    loglik = np.zeros((T))  # np.log-likelihood
    K_a = np.zeros((T, n, p))  # analysed Kalman gain
    x_a[0, :] = x0
    P_a[0, :, :] = P0

    # apply the Kalman filter
    for k in range(1, T):
        # prediction step
        x_f[k, :] = M[k, :, :] @ x_a[k - 1, :]
        P_f[k, :, :] = M[k, :, :] @ P_a[k - 1, :, :] @ M[k, :, :].T + Q[k, :, :]

        # Kalman gain
        K_a[k, :, :] = P_f[k, :, :] @ H.T @ np.linalg.inv(H @ P_f[k, :, :] @ H.T + R)

        # update step
        x_a[k, :] = x_f[k, :] + K_a[k, :, :] @ (y[k, :] - H @ x_f[k, :])
        P_a[k, :, :] = P_f[k, :] - K_a[k, :, :] @ H @ P_f[k, :, :]

        # stock the np.log-likelihood
        loglik[k] = -0.5 * (
            (y[k, :] - H @ x_f[k, :]).T
            @ np.linalg.inv(H @ P_f[k, :, :] @ H.T + R)
            @ (y[k, :] - H @ x_f[k, :])
        ) - 0.5 * (
            n * np.log(2 * np.pi) + np.log(np.linalg.det(H @ P_f[k, :, :] @ H.T + R))
        )

    return x_f, P_f, x_a, P_a, loglik, K_a


def Kalman_smoother_time_dependent(
    y: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
    M: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Apply the linear and Gaussian Kalman smooother to estimate the state of a
    time-dependent system. Thus the provided Observation.

    This function applies the Kalman smoother to a time-dependent system with linear dynamics and Gaussian noise.

    The underlying Kalman equations are:
    x(t) = M x(t+dt) + Q
    y(t) = H x(t) + R

    Parameters
    ----------
    y : array-like, shape (T, p)
        Observations of the system state observed over time.
    x0 : array-like, shape (n,)
        Initial estimate of the system state.
    P0 : array-like, shape (n, n)
        Initial estimate of the error covariance matrix of the system state.
    M : array-like, shape (T, n, n)
        Time-varying state transition matrix.
    Q : array-like, shape (T, n, n)
        Time-varying process noise covariance matrix.
    H : array-like, shape (p, n)
        Observation matrix that maps the state space to the observation space.
    R : array-like, shape (p, p)
        Observation noise covariance matrix.

    Returns
    -------
    x_f : array-like, shape (T, n)
        Forecasted state estimates.
    P_f : array-like, shape (T, n, n)
        Forecasted error covariance matrices of the state estimates.
    x_a : array-like, shape (T, n)
        Updated state estimates after assimilating the observations.
    P_a : array-like, shape (T, n, n)
        Updated error covariance matrices of the state estimates after assimilating the observations.
    loglik : array-like, shape (T,)
        Log-likelihood values of the observations.
    K_a : array-like, shape (T, n, p)
        Kalman gain matrices used in the analysis step.
    x_s : array, shape (T, n)
        Smoothed updated state estimates after assimilating the observations.
    P_s : array, shape (T, n, n)
        Smoothed error covariance matrices of the state estimates after assimilating the observations.
    P_s_lag : array, shape (T, n, n)
        Smoothed lagged error covariance matrices of the state estimates after assimilating the observations.
    """

    # shapes
    n = len(x0)
    T, p = np.shape(y)

    # Kalman initialization
    x_s = np.zeros((T, n))  # smoothed state
    P_s = np.zeros((T, n, n))  # smoothed error covariance matrix
    P_s_lag = np.zeros((T - 1, n, n))  # smoothed lagged error covariance matrix

    # apply the Kalman filter
    x_f, P_f, x_a, P_a, loglik, K_a = Kalman_filter_time_dependent(
        y, x0, P0, M, Q, H, R
    )

    # apply the Kalman smoother
    x_s[-1, :] = x_a[-1, :]
    P_s[-1, :, :] = P_a[-1, :, :]
    for k in range(T - 2, -1, -1):
        K = P_a[k, :, :] @ M[k, :, :].T @ np.linalg.inv(P_f[k + 1, :, :])
        x_s[k, :] = x_a[k, :] + K @ (x_s[k + 1, :] - x_f[k + 1, :])
        P_s[k, :, :] = P_a[k, :, :] + K @ (P_s[k + 1, :, :] - P_f[k + 1, :, :]) @ K.T

    for k in range(0, T - 1):
        A = (np.eye(n) - K_a[k + 1, :, :] @ H) @ M[k, :, :] @ P_a[k, :, :]
        B = (P_s[k + 1, :, :] - P_a[k + 1, :, :]) @ np.linalg.inv(P_a[k + 1, :, :])
        P_s_lag[k, :, :] = A + B @ A

    return x_f, P_f, x_a, P_a, x_s, P_s, loglik, P_s_lag


def Kalman_SEM_time_dependent(
    x: np.ndarray,
    y: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    nb_iter_SEM: int,
    sigma: float = 100,
    seed: int = 11,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Apply the Kalman Stochastic Expectation Maximization alogrihtm to estimate
    the state of a time-dependent system. This function applies the Kalman
    Stochastic Expectation Maximization alogrihtm to a time-dependent system
    with linear dynamics and Gaussian noise.

    To account for the time dependency of the system, a local linear regression is used to update the estimates of the
    - true state transition matrix M and
    - true process noise covariance matrix Q.

    A Gaussian 1D kernel with variance ``sigma`` is used.
    The kernel is applied in the ``time`` space along the first axis of the input array x and y.
    Therefore it does not represent a multidimensional kernel or a method similar to using Analogs.
    Note that the sigma is unitless and is measurent in index-positions of the array.

    The underlying Kalman equations are:
    x(t) = M x(t+dt) + Q
    y(t) = H x(t) + R

    Parameters
    ----------
    x : array-like, shape (T, n)
        System state over time period of length T, having n states.
    y : array-like, shape (T, p)
        Observations of the system state observed over time.
    H : array-like, shape (p, n)
        First estimation of the observation matrix that maps the state space to the observation space.
    R : array-like, shape (p, p)
        First estimation of the observation noise covariance matrix.
    nb_iter_SEM : int
        Number of iterations that shall be performed for the alogithm.
    sigma : float
        Standard deviation as the sqrt(variance) of the Gaussian distribution to create the 1D kernel used for the local linear regression.
        Note that the sigma is unitless and is measurent in index-positions of the array.
    seed : int, optional
        Seed for the NumPy random number generator. By providing a seed,
        you can ensure reproducibility of the random numbers generated.
        If not specified, a default seed will be used 11.

    Returns
    -------
    x_s : array, shape (T, n)
        Smoothed updated state estimates after assimilating the observations.
        This represents the state (x) of the system after the final iteration of the algorithm.
    P_s : array, shape (T, n, n)
        Smoothed error covariance matrices of the state estimates after assimilating the observations.
        This represents the error covariance matrices as a function of time after the final iteration of the algorithm.
    M : array, shape (T, n, n)
        Estimation of the true time-varying state transition matrix M after the final iteration of the algorithm.
    tab_loglik : array, shape (nb_iter_SEM)
        Sum of the log-likelihod over time dimension for each iteration of the algorithm.
    x_out : array, shape (T, n)
        Simulated the new state based on a multivariate normal distribution which uses
        x_s as mean and P_s as Covariance matrix.
    x_f : array, shape (T, n)
        Forecasted state estimates after assimilating the observations by the used Kalman Filter, after the final iteration of the algorithm.
    Q : array shape (T, p, p)
        Estimation of the true time-varying process noise covariance matrix Q after the final iteration of the algorithm.
    """
    # verify that non nans in the data
    x_nans = np.sum(np.isnan(x)) == 0
    y_nans = np.sum(np.isnan(y)) == 0
    if not x_nans or not y_nans:
        raise NotImplementedError(
            "It seems that the provided 'x' or 'y' arrays contain nan values.\nThis is not supported yet!"
        )

    # fix the seed
    np.random.seed(seed=seed)

    # copy x
    x_out = x.copy()

    # shapes
    T = np.shape(x_out)[0]
    n = np.shape(x_out)[1]

    # tab to store the np.log-likelihood
    tab_loglik = []

    M = np.zeros((T, n, n))
    Q = np.zeros_like(M)

    # loop on the SEM iterations
    for i in tqdm(np.arange(0, nb_iter_SEM)):
        # Use a local linear regression to compute M for each timestep.
        # This is done by using a 1D gaussian kernel centered at the corresponding timestep.
        for idx in range(0, T):
            # Get the sample weights as 1D gaussian kernel based on sigma
            sample_weight = gaussian_kernel_1D(
                x_out[:-1,], idx, axis=0, sigma=sigma, same_output_shape=False
            )
            # Kalman parameters for each timestep
            reg = LinearRegression(fit_intercept=False).fit(
                x_out[:-1,], x_out[1:,], sample_weight=sample_weight
            )
            M[idx,] = reg.coef_
            Q[idx,] = np.cov((x_out[1:,] - reg.predict(x_out[:-1,])).T)
        # R   = np.cov(y.T - H @ x.T)

        # Kalman initialization
        if i == 0:
            x0 = np.zeros(n)
            P0 = np.eye(n)
        else:
            x0 = x_s[0, :]
            P0 = P_s[
                0,
                :,
                :,
            ]

        # apply the Kalman smoother
        x_f, P_f, x_a, P_a, x_s, P_s, loglik, P_s_lag = Kalman_smoother_time_dependent(
            y, x0, P0, M, Q, H, R
        )

        # store the np.log-likelihod
        tab_loglik = np.append(tab_loglik, sum(loglik))

        # simulate the new x
        for k in range(len(x_s)):
            x_out[k, :] = np.random.multivariate_normal(x_s[k, :], P_s[k, :, :])

    return x_s, P_s, M, tab_loglik, x_out, x_f, Q
