import numpy as np
import scipy as sp


def my_mean(x):
    """Calculate the mean of a numpy array using np.mean
    This functions only purpose is to check the CI workflows of the repository.

    Parameters
    ----------
    x: np.ndarray
        Array for which the mean shall be calculated.

    Returns
    -------
    np.ndarray
        Mean of input array x.
        ndim=1

    """

    return np.mean(x)


def gaussian_weights(x, y, axis=0, alpha=0.2):
    """Creates a Gaussian weights for a 2D-array x centered at positions given by y.

    Parameters
    ----------
    x: np.ndarray
        Array of dimension (n, m).
        ndim should be 2
    y: np.ndarray
        Array of dimension (n, ).
        ndim should be 2 but 1 is also accepted.
        if ndim is 1
    alpha: float
        Alpha of the gaus distribution.
        Default to 0.2

    Returns
    -------
    np.ndarray
        gaussian weights
        dimension as x

    """
    try:
        assert np.array_equal(np.ndim(x), np.ndim(y))
    except Exception:
        if axis == 0:
            y = y[:, np.newaxis]
        elif axis == 1:
            y = y[np.newaxis, :]
    # apply the Gaussian kernel
    weights = np.exp((-((x - y) ** 2)) / (2 * alpha**2))

    # normalize the weights
    weights = weights / np.sum(weights)

    # return the weights
    return weights
