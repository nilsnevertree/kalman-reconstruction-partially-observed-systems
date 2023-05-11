""" Timedependent verions of Kalman.py:
Apply the linear and Gaussian Kalman filter and smoother with timedependent Model M and Model Error Q """

import numpy as np
import scipy as sp

from sklearn.linear_model import LinearRegression
from tqdm import tqdm


def Kalman_filter_time_dependent(y, x0, P0, M, Q, H, R):
    """Apply the linear and Gaussian Kalman filter."""

    # shapes
    n = len(x0)
    T, p = np.shape(y)

    # Kalman initialization
    x_f = np.empty((T, n))  # forecast state
    P_f = np.empty((T, n, n))  # forecast error covariance matrix
    x_a = np.empty((T, n))  # analysed state
    P_a = np.empty((T, n, n))  # analysed error covariance matrix
    loglik = np.empty((T))  # np.log-likelihood
    K_a = np.empty((T, n, p))  # analysed Kalman gain
    x_a[0, :] = x0
    P_a[0, :, :] = P0

    # apply the Kalman filter
    for k in range(1, T):
        # prediction step
        x_f[k, :] = M[k, :, :] @ x_a[k - 1, :]
        P_f[k, :, :] = M[k, :, :] @ P_a[k - 1, :, :] @ M[k, :, :].T + Q

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


def Kalman_smoother_time_dependent(y, x0, P0, M, Q, H, R):
    """Apply the linear and Gaussian Kalman smoother."""

    # shapes
    n = len(x0)
    T, p = np.shape(y)

    # Kalman initialization
    x_s = np.empty((T, n))  # smoothed state
    P_s = np.empty((T, n, n))  # smoothed error covariance matrix
    P_s_lag = np.empty((T - 1, n, n))  # smoothed lagged error covariance matrix

    # apply the Kalman filter
    x_f, P_f, x_a, P_a, loglik, K_a = Kalman_filter_time_dependent(y, x0, P0, M, Q, H, R)

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


def Kalman_SEM__time_dependent(x, y, H, R, nb_iter_SEM):  # , x_t, t):
    """Apply the stochastic expectation-maximization algorithm."""

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
            x0 = np.empty(n)
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
