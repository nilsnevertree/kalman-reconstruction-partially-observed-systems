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


def gaussian_weights_2D(x, y, axis=0, alpha=0.2):
    """Creates a Gaussian weights for a 2D-array x centered at positions given by y.
    The weights will be computed along the specified axis.

    Parameters
    ----------
    x: np.ndarray
        Array of dimension (n, m).
        x.ndim should be 2.
    y: np.ndarray
        Array of dimension (n, ).
        y.ndim should be 2 or 1.
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


def broadcast_along_axis_as(b, a, axis):
    """
    Based on the answer at https://stackoverflow.com/a/62655664/16372843
    """
    # shape check
    if axis >= a.ndim:
        raise np.AxisError(axis, a.ndim)
    if b.ndim != 1:
        raise ValueError(f"ndim of 'b' : {b.ndim} must be 1")

    if a.shape[axis] != b.size:
        raise ValueError("Length of 'b' must be the same as a.shape along the axis")

    # np.broadcast_to puts the new axis as the last axis, so
    # we swap the given axis with the last one, to determine the
    # corresponding array shape. np.swapaxes only returns a view
    # of the supplied array, so no data is copied unnecessarily.
    shape = np.swapaxes(a, a.ndim - 1, axis).shape

    # Broadcast to an array with the shape as above. Again,
    # no data is copied, we only get a new look at the existing data.
    b_brc = np.broadcast_to(b, shape)

    # Swap back the axes. As before, this only changes our "point of view".
    b_brc = np.swapaxes(b_brc, a.ndim - 1, axis)
    return b_brc


def gaussian_kernel_1D(x, center_idx, axis=0, sigma=0.2, same_output_shape=False):
    """Creates a Gaussian weights for a N dimensional array x centered at index y along specified axis.

    Parameters
    ----------
    x: np.ndarray
        Array of dimension N with length l along provided axis.
    center_idx: int
        Index of the center of the gaussian kernel along provided axis.
    axis: int
        Axis along which the weights shall be computed.
    sigma: float
        Alpha of the gaus distribution.
        Default to 0.2

    Returns
    -------
    np.ndarray
        1D array of length x.shape[axis] containing the gaussian kernel.

    """

    index_array = np.arange(x.shape[axis])

    # apply the Gaussian kernel
    kernel = np.exp((-((index_array - center_idx) ** 2)) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    # normalize the weights
    if not same_output_shape:
        return kernel
    else:
        return broadcast_along_axis_as(kernel, x, axis=axis)
