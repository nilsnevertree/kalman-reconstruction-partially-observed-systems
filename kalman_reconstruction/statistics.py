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


def broadcast_along_axis_as(x, y, axis):
    """Broadcasts 1D array x to an array of same shape as y, containing the given axis.
    The length of x need to be the same as the length of y along the given axis.
    Note that this is a broadcast_to, so the return is a view on x.
    Based on the answer at https://stackoverflow.com/a/62655664/16372843

    Parameters
    ----------
    x: np.ndarray
        Array of dimension 1 which should be broadcasted for a specific axis.
    x: np.ndarray
        Array of dimension 1 which should be broadcasted for a specific axis.
    axis: int
        Axis along which the arrays allign.

    Returns
    -------
    np.ndarray
        Array containing values along provided axis as x but with shape y.
        Note that this is a broadcast_to, so the return is a view on x.
    """
    # shape check
    if axis >= y.ndim:
        raise np.AxisError(axis, y.ndim)
    if x.ndim != 1:
        raise ValueError(f"ndim of 'x' : {x.ndim} must be 1")
    if x.size != y.shape[axis]:
        raise ValueError(
            f"Length of 'x' must be the same as y.shape along the axis. But found {x.size}, {y.shape[axis]}, axis= {axis}"
        )

    # np.broadcast_to puts the new axis as the last axis, so
    # we swap the given axis with the last one, to determine the
    # corresponding array shape. np.swapaxes only returns a view
    # of the supplied array, so no data is copied unnecessarily.
    shape = np.swapaxes(y, y.ndim - 1, axis).shape

    # Broadcast to an array with the shape as above. Again,
    # no data is copied, we only get a new look at the existing data.
    res = np.broadcast_to(x, shape)

    # Swap back the axes. As before, this only changes our "point of view".
    res = np.swapaxes(res, y.ndim - 1, axis)
    return res


def gaussian_kernel_1D(x, center_idx, axis=0, sigma=10, same_output_shape=False):
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
        sigma of the gaus distribution.
        Default to 0.2
    same_output_shape : bool
        Sets if the output array should be of shape.
        - Output array is 1D array if False.
        - Output array of same shape as x if True.
          Then the weights will be along the provided axis

    Returns
    -------
    np.ndarray
        Array containing the weights of the kernel.
        If output is 1D, along this axis.
        If output in ND, along provided axis.
        See also 'same_output_shape'.
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
