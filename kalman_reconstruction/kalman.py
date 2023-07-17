""" kalman.py: Apply the linear and Gaussian Kalman filter and smoother

__author__ = "Pierre Tandeo"
__version__ = "0.1"
__date__ = "2022-03-09"
__maintainer__ = "Pierre Tandeo"
__email__ = "pierre.tandeo@imt-atlantique.fr"

"""

import numpy as np
import scipy as sp

from sklearn.linear_model import LinearRegression
from tqdm import tqdm


def Kalman_filter(y, x0, P0, M, Q, H, R):
    """
    Apply the linear and Gaussian Kalman filter to estimate the state of a
    system.

    This function applies the Kalman filter to a system with linear dynamics and Gaussian noise.

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
    M : array-like, shape (n, n)
        State transition matrix.
    Q : array-like, shape (n, n)
        Process noise covariance matrix.
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
        x_f[k, :] = M @ x_a[k - 1, :]
        P_f[k, :, :] = M @ P_a[k - 1, :, :] @ M.T + Q

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


def Kalman_smoother(y, x0, P0, M, Q, H, R):
    """
    Apply the linear and Gaussian Kalman smooother to estimate the state of a
    system.

    This function applies the Kalman smoother to a system with linear dynamics and Gaussian noise.

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
    M : array-like, shape (n, n)
        State transition matrix.
    Q : array-like, shape (n, n)
        Process noise covariance matrix.
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
    x_f, P_f, x_a, P_a, loglik, K_a = Kalman_filter(y, x0, P0, M, Q, H, R)

    # apply the Kalman smoother
    x_s[-1, :] = x_a[-1, :]
    P_s[-1, :, :] = P_a[-1, :, :]
    for k in range(T - 2, -1, -1):
        K = P_a[k, :, :] @ M.T @ np.linalg.inv(P_f[k + 1, :, :])
        x_s[k, :] = x_a[k, :] + K @ (x_s[k + 1, :] - x_f[k + 1, :])
        P_s[k, :, :] = P_a[k, :, :] + K @ (P_s[k + 1, :, :] - P_f[k + 1, :, :]) @ K.T

    for k in range(0, T - 1):
        A = (np.eye(n) - K_a[k + 1, :, :] @ H) @ M @ P_a[k, :, :]
        B = (P_s[k + 1, :, :] - P_a[k + 1, :, :]) @ np.linalg.inv(P_a[k + 1, :, :])
        P_s_lag[k, :, :] = A + B @ A

    return x_f, P_f, x_a, P_a, x_s, P_s, loglik, P_s_lag


def Kalman_EM(y, xb, B, M, Q, H, R, nb_iter_EM):
    """Apply the expectation-maximization algorithm."""

    # Kalman initialization
    x0 = xb[0, :]
    P0 = B

    # apply the Kalman smoother
    x_f, P_f, x_a, P_a, x_s, P_s, loglik = Kalman_smoother(y, x0, P0, M, Q, H, R)

    # copy x
    # x_s = xb.copy()

    # shapes
    n = np.shape(xb)[1]

    # tab to store the np.log-likelihood
    tab_loglik = []

    # loop on the SEM iterations
    for i in tqdm(np.arange(0, nb_iter_EM)):
        # Kalman parameters
        reg = LinearRegression(fit_intercept=False).fit(x_s[:-1,], x_s[1:,])
        M = reg.coef_
        Q = np.cov((x_s[1:,] - reg.predict(x_s[:-1,])).T)
        # R   = np.cov(y.T - H @ x.T)

        # Kalman initialization
        x0 = x_s[0, :]
        P0 = P_s[
            0,
            :,
            :,
        ]

        # apply the Kalman smoother
        x_f, P_f, x_a, P_a, x_s, P_s, loglik = Kalman_smoother(y, x0, P0, M, Q, H, R)

        # store the np.log-likelihod
        tab_loglik = np.append(tab_loglik, sum(loglik))

    return x_s, P_s, M, tab_loglik


def Kalman_SEM(x, y, H, R, nb_iter_SEM):  # , x_t, t):
    """
    Apply the Kalman Stochastic Expectation Maximization alogrihtm to estimate
    the state of a system. This function applies the Kalman Stochastic
    Expectation Maximization alogrihtm to a system with linear dynamics and
    Gaussian noise.

    A global linear regression is used to update the estimates of the
    - true state transition matrix M and
    - true process noise covariance matrix Q.

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
    M : array, shape (n, n)
        Estimation of the true state transition matrix M after the final iteration of the algorithm.
    tab_loglik : array, shape (nb_iter_SEM)
        Sum of the log-likelihod over time dimension for each iteration of the algorithm.
    x_out : array, shape (T, n)
        Simulated the new state based on a multivariate normal distribution which uses
        x_s as mean and P_s as Covariance matrix.
    x_f : array, shape (T, n)
        Forecasted state estimates after assimilating the observations by the used Kalman Filter, after the final iteration of the algorithm.
    Q : array shape (T, p, p)
        Estimation of the true process noise covariance matrix Q after the final iteration of the algorithm.
    """
    # verify that non nans in the data
    x_nans = np.sum(np.isnan(x)) == 0
    y_nans = np.sum(np.isnan(y)) == 0
    if not x_nans or not y_nans:
        raise NotImplementedError(
            "It seems that the provided 'x' or 'y' arrays contain nan values.\nThis is not supported yet!"
        )

    # fix the seed
    np.random.seed(11)

    # copy x
    x_out = x.copy()

    # shapes
    n = np.shape(x_out)[1]

    # tab to store the np.log-likelihood
    tab_loglik = []

    # loop on the SEM iterations
    for i in tqdm(np.arange(0, nb_iter_SEM)):
        # Kalman parameters
        reg = LinearRegression(fit_intercept=False).fit(x_out[:-1,], x_out[1:,])
        M = reg.coef_
        Q = np.cov((x_out[1:,] - reg.predict(x_out[:-1,])).T)
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
        x_f, P_f, x_a, P_a, x_s, P_s, loglik, P_s_lag = Kalman_smoother(
            y, x0, P0, M, Q, H, R
        )

        # store the np.log-likelihod
        tab_loglik = np.append(tab_loglik, sum(loglik))

        # simulate the new x
        for k in range(len(x_s)):
            x_out[k, :] = np.random.multivariate_normal(x_s[k, :], P_s[k, :, :])

    return x_s, P_s, M, tab_loglik, x_out, x_f, Q


def Kalman_SEM_full_output(x, y, H, R, nb_iter_SEM):  # , x_t, t):
    """
    Apply the Kalman Stochastic Expectation Maximization alogrihtm to estimate
    the state of a system. This function applies the Kalman Stochastic
    Expectation Maximization alogrihtm to a system with linear dynamics and
    Gaussian noise. Gives the results for each iteration in first axis!

    A global linear regression is used to update the estimates of the
    - true state transition matrix M and
    - true process noise covariance matrix Q.

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
    seed : int, optional
        Seed for the NumPy random number generator. By providing a seed,
        you can ensure reproducibility of the random numbers generated.
        If not specified, a default seed will be used 11.

    Returns
    -------
    x_s : array, shape (nb_iter_SEM, T, n)
        Smoothed updated state estimates after assimilating the observations.
        This represents the state (x) of the system after the final iteration of the algorithm.
    P_s : array, shape (nb_iter_SEM, T, n, n)
        Smoothed error covariance matrices of the state estimates after assimilating the observations.
        This represents the error covariance matrices as a function of time after the final iteration of the algorithm.
    M : array, shape (nb_ier_SEM, n, n)
        Estimation of the true state transition matrix M after the final iteration of the algorithm.
    tab_loglik : array, shape (nb_iter_SEM)
        Sum of the log-likelihod over time dimension for each iteration of the algorithm.
    x_out : array, shape (nb_iter_SEM, T, n)
        Simulated the new state based on a multivariate normal distribution which uses
        x_s as mean and P_s as Covariance matrix.
    x_f : array, shape (nb_iter_SEM, T, n)
        Forecasted state estimates after assimilating the observations by the used Kalman Filter, after the final iteration of the algorithm.
    Q : array shape (T, p, p)
        Estimation of the true process noise covariance matrix Q after the final iteration of the algorithm.
    """
    # verify that non nans in the data
    x_nans = np.sum(np.isnan(x)) == 0
    y_nans = np.sum(np.isnan(y)) == 0
    if not x_nans or not y_nans:
        raise NotImplementedError(
            "It seems that the provided 'x' or 'y' arrays contain nan values.\nThis is not supported yet!"
        )

    # fix the seed
    np.random.seed(11)

    # copy x
    x_out = x.copy()

    # shapes
    n = np.shape(x_out)[1]
    T = np.shape(x_out)[0]

    # create array for the return
    x_s_full = np.zeros((nb_iter_SEM, T, n))
    P_s_full = np.zeros((nb_iter_SEM, T, n, n))
    x_out_full = np.zeros_like(x_s_full)
    x_f_full = np.zeros_like(x_s_full)
    M_full = np.zeros((nb_iter_SEM, n, n))
    Q_full = np.zeros_like(M_full)
    # tab to store the np.log-likelihood
    tab_loglik = []

    # loop on the SEM iterations
    for i in tqdm(np.arange(0, nb_iter_SEM)):
        # Kalman parameters
        reg = LinearRegression(fit_intercept=False).fit(x_out[:-1,], x_out[1:,])
        M = reg.coef_
        Q = np.cov((x_out[1:,] - reg.predict(x_out[:-1,])).T)
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
        x_f, P_f, x_a, P_a, x_s, P_s, loglik, P_s_lag = Kalman_smoother(
            y, x0, P0, M, Q, H, R
        )

        # store the np.log-likelihod
        tab_loglik = np.append(tab_loglik, sum(loglik))

        # simulate the new x
        for k in range(len(x_s)):
            x_out[k, :] = np.random.multivariate_normal(x_s[k, :], P_s[k, :, :])

        x_s_full[i, :] = x_s
        P_s_full[i, :] = P_s
        x_out_full[i, :] = x_out
        x_f_full[i, :] = x_f
        M_full[i, :] = M
        Q_full[i, :] = Q

    return x_s_full, P_s_full, M_full, tab_loglik, x_out_full, x_f_full, Q_full


def Kalman_SEM_bis(x, y, H, R, nb_iter_SEM, M_init, Q_init):
    """Apply the stochastic expectation-maximization algorithm."""

    # fix the seed
    np.random.seed(11)

    # copy x
    x_out = x.copy()

    # shapes
    n = np.shape(x_out)[1]
    T, p = np.shape(y)

    # tab to store the np.log-likelihood
    tab_loglik = []

    # loop on the SEM iterations
    for i in tqdm(np.arange(0, nb_iter_SEM)):
        # Kalman initialization
        if i == 0:
            x0 = np.zeros(n)
            P0 = np.eye(n)
            M = M_init
            Q = Q_init
        else:
            x0 = x_s[0, :]
            P0 = P_s[
                0,
                :,
                :,
            ]

        # apply the Kalman smoother
        x_f, P_f, x_a, P_a, x_s, P_s, loglik, P_s_lag = Kalman_smoother(
            y, x0, P0, M, Q, H, R
        )

        # update the Kalman parameters
        A = np.zeros((n, n))
        for k in np.arange(0, T - 1):
            A += P_s[k, :, :] + np.array([x_s[k, :]]).T @ np.array([x_s[k, :]])
        B = np.zeros((n, n))
        for k in np.arange(0, T - 1):
            B += P_s_lag[k, :, :] + np.array([x_s[k + 1, :]]).T @ np.array([x_s[k, :]])
        C = np.zeros((n, n))
        for k in np.arange(0, T - 1):
            C += P_s[k + 1, :, :] + np.array([x_s[k + 1, :]]).T @ np.array(
                [x_s[k + 1, :]]
            )
        M = B @ np.linalg.inv(A)
        Q = (C - M @ B.T) / (T - 1)

        # store the np.log-likelihod
        tab_loglik = np.append(tab_loglik, sum(loglik))

        # simulate the new x
        for k in range(len(x_s)):
            x_out[k, :] = np.random.multivariate_normal(x_s[k, :], P_s[k, :, :])

    return x_s, P_s, M, tab_loglik, x_out, x_f


def ensemble_Kalman_filter(y, x0, P0, m, Q, H, R, Ne):
    """Apply the ensemble Kalman filter (stochastic version)."""

    # shapes
    n = np.shape(x0)[1]  # n = len(x0)
    T, p = np.shape(y)

    # Kalman initialization
    x_f = np.zeros((T, n))  # forecast state
    P_f = np.zeros((T, n, n))  # forecast error covariance matrix
    x_a = np.zeros((T, n))  # analysed state
    P_a = np.zeros((T, n, n))  # analysed error covariance matrix
    loglik = np.zeros((T))  # np.log-likelihood
    x_f_tmp = np.zeros((n, Ne))  # members of the forecast
    y_f_tmp = np.zeros((p, Ne))  # members of the perturbed observations
    x_a_tmp = np.zeros((n, Ne))  # members of the analysis
    x_a_tmp = np.random.multivariate_normal(np.squeeze(x0), P0, Ne).T
    x_a[0, :] = np.mean(x_a_tmp, 1)
    P_a[0, :, :] = np.cov(x_a_tmp)

    # apply the ensemble Kalman filter
    for k in range(1, T):
        # prediction step
        for i in range(Ne):
            x_f_tmp[:, i] = m(x_a_tmp[:, i]) + np.random.multivariate_normal(
                np.zeros(n), Q
            )
            y_f_tmp[:, i] = H @ x_f_tmp[:, i] + np.random.multivariate_normal(
                np.zeros(p), R
            )
        P_f[k, :, :] = np.cov(x_f_tmp)

        # Kalman gain
        K = P_f[k, :, :] @ H.T @ np.linalg.inv(H @ P_f[k, :, :] @ H.T + R)

        # update step
        if sum(np.isfinite(y[k, :])) > 0:  # if observations are available
            for i in range(Ne):
                x_a_tmp[:, i] = x_f_tmp[:, i] + K @ (y[k, :] - y_f_tmp[:, i])
            P_a[k, :, :] = np.cov(x_a_tmp)
        else:
            x_a_tmp = x_f_tmp
            P_a[k, :, :] = P_f[k, :, :]
        x_a[k, :] = np.mean(x_a_tmp, 1)

        # stock the np.log-likelihood
        loglik[k] = -0.5 * (
            (y[k, :] - H @ x_f[k, :]).T
            @ np.linalg.inv(H @ P_f[k, :, :] @ H.T + R)
            @ (y[k, :] - H @ x_f[k, :])
        ) - 0.5 * (
            n * np.log(2 * np.pi) + np.log(np.linalg.det(H @ P_f[k, :, :] @ H.T + R))
        )

    return x_f, P_f, x_a, P_a, loglik
