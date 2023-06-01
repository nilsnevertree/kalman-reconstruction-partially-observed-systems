import numpy as np
import pytest
import xarray as xr

from numpy.testing import assert_almost_equal

from kalman_reconstruction.statistics import (
    assert_ordered_subset,
    coverage,
    coverage_prob,
    gaussian_kernel_1D,
    my_mean,
    ordered_like,
    xarray_coverage_prob,
    xarray_RMSE,
)


# Test data

x = [0, 1, 2, 3]
y = [2, 4]
data = [[1, 2, 3, 4], [4, 3, np.nan, 1]]
weights = [[1, 1, 1, 1], [2, 2, 2, 2]]


@pytest.fixture
def example_array_01():
    data = np.array(
        [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, np.nan, -1.0]],
    )
    return data


@pytest.fixture
def example_array_02():
    data = np.array(
        [[-2.0, 2.0, 3.0, 6.0], [4.0, 10.0, np.nan, -10.0]],
    )
    return data


@pytest.fixture
def example_covariance_01():
    """
    Covariance matrix.

    Note that the values in teh array are the standard deviations
    """
    data = np.array(
        [[2.0, 2.0, 2.0, 2.0], [np.nan, 1.0, np.nan, 3.0]],
    )
    data = data**2
    return data


@pytest.fixture
def coverage_01():
    """Coverage result using 1 standard deviations."""
    data = np.array(
        [[False, True, True, True], [False, False, False, False]],
    )
    return data


@pytest.fixture
def coverage_probability_01():
    """Coverage result using 1 standard deviations."""
    data = np.array(3 / 8)
    return data


@pytest.fixture
def mean_01():
    data = np.array(
        [2.5, np.nan],
    )
    return data


@pytest.fixture
def mean_all() -> np.ndarray:
    data = np.array([np.nan])
    return data


def test_my_mean_nan(example_array_01, mean_01):
    result = my_mean(example_array_01, axis=1)
    np.testing.assert_allclose(mean_01, result)


def test_my_mean(example_array_01, mean_all):
    result = my_mean(example_array_01)
    np.testing.assert_allclose(mean_all, result)


def test_coverage(
    example_array_01, example_array_02, example_covariance_01, coverage_01
):
    result1 = coverage(
        x=example_array_01, P=example_covariance_01, y=example_array_02, stds=1
    )
    result2 = coverage(
        x=example_array_02, P=example_covariance_01, y=example_array_01, stds=1
    )

    np.testing.assert_allclose(result1, result2)
    np.testing.assert_allclose(coverage_01, result1)
    np.testing.assert_allclose(coverage_01, result2)


def test_coverage_prob(
    example_array_01, example_array_02, example_covariance_01, coverage_probability_01
):
    result1 = coverage_prob(
        x=example_array_01, P=example_covariance_01, y=example_array_02, stds=1
    )
    result2 = coverage_prob(
        x=example_array_02, P=example_covariance_01, y=example_array_01, stds=1
    )

    np.testing.assert_allclose(result1, result2)
    np.testing.assert_allclose(coverage_probability_01, result1)
    np.testing.assert_allclose(coverage_probability_01, result2)


# @pytest.fixture
def example_DataArray_01():
    data = np.array(
        [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, np.nan, -1.0]],
    )
    data = xr.DataArray(data=data, dims=["x", "y"])
    return data


# @pytest.fixture
def example_DataArray_02():
    data = np.array(
        [[-2.0, 2.0, 3.0, 6.0], [4.0, 10.0, np.nan, -10.0]],
    )
    data = xr.DataArray(data=data, dims=["x", "y"])
    return data


# @pytest.fixture
def example_covariance_DataArray():
    data = np.array(
        [[2.0, 2.0, 2.0, 2.0], [np.nan, 1.0, np.nan, 3.0]],
    )
    data = data**2
    data = xr.DataArray(data=data, dims=["x", "y"])
    return data


# @pytest.fixture
def coverage_DataArray():
    """Coverage result using 1 standard deviations."""
    data = np.array(
        [[False, True, True, True], [False, False, False, False]],
    )
    data = xr.DataArray(data=data, dims=["x", "y"])
    return data


# @pytest.fixture
def coverage_probability_DataArray():
    """Coverage result using 1 standard deviations."""
    data = np.array([3 / 4, 0])
    data = xr.DataArray(data=data, dims=["y"])
    return data


def test_xarray_coverage(
    x=example_DataArray_01(),
    y=example_DataArray_02(),
    P=example_covariance_DataArray(),
    expected=coverage_DataArray(),
):
    result1 = coverage(x=x, P=P, y=y, stds=1)
    result2 = coverage(x=y, P=P, y=x, stds=1)

    np.testing.assert_allclose(result1, result2)
    np.testing.assert_allclose(expected, result1)
    np.testing.assert_allclose(expected, result2)


def test_xarray_coverage_probability(
    x=example_DataArray_01(),
    y=example_DataArray_02(),
    P=example_covariance_DataArray(),
    expected=coverage_probability_DataArray(),
):
    result1 = xarray_coverage_prob(x=x, P=P, y=y, stds=1, dim="y")
    result2 = xarray_coverage_prob(x=y, P=P, y=x, stds=1, dim="y")

    assert result1.dims == ("x",)
    assert result2.dims == ("x",)
    np.testing.assert_allclose(result1, result2)
    np.testing.assert_allclose(expected, result1)
    np.testing.assert_allclose(expected, result2)


def test_xarray_RMSE(x=example_DataArray_01(), y=example_DataArray_02()):
    # [[1., 2., 3., 4.], [4., 3., np.nan, -1.]],
    # [[-2., 2., 3., 6.], [4., 10., np.nan, -10.]],
    print(x, y)
    print((x - y) ** 2)
    result = xarray_RMSE(x=x, y=y, dim="y")
    diff = np.array([[9.0, 0.0, 0.0, 4.0], [0.0, 49.0, np.nan, 81.0]])
    expected_mean = np.array([13 / 4, 130 / 3])
    expected = np.sqrt(expected_mean)
    assert_almost_equal(expected, result)


def test_orderd_like():
    a = ["x2", "x3"]
    b = ["x2", "z1", "x3", "z2"]
    resultb = ordered_like(b, a)
    resulta = ordered_like(a, b)
    assert resultb == ["x2", "x3", "z1", "z2"]
    assert resulta == ["x2", "x3"]


def test_assert_ordered_subset():
    a = ["x2", "x3"]
    b = ["x2", "x1", "x3", "x4"]
    assert None == assert_ordered_subset(a, b)
    # No exception raised

    a = ["x2", "x3"]
    b = ["x3", "x1", "x2", "x4"]
    with pytest.raises(AssertionError, match="a is not ordered like of b"):
        assert_ordered_subset(a, b)

    a = ["a2", "x3"]
    b = ["x3", "x1", "x2", "x4"]
    with pytest.raises(AssertionError, match="a is not a subset of b"):
        assert_ordered_subset(a, b)


def test_gaussian_kernel_1D(x=example_DataArray_01()):
    result1 = gaussian_kernel_1D(
        x=x, center_idx=1, axis=1, sigma=np.exp(1), same_output_shape=False
    )
    result2 = gaussian_kernel_1D(
        x=x, center_idx=2, axis=1, sigma=np.exp(1), same_output_shape=False
    )
    assert_almost_equal(np.array([0.2573151, 0.2753297, 0.2573151, 0.2100401]), result1)
    assert_almost_equal(np.array([0.2100401, 0.2573151, 0.2753297, 0.2573151]), result2)
    assert np.shape(result1) == (4,)
    assert np.shape(result2) == (4,)
    result1 = gaussian_kernel_1D(
        x=x, center_idx=1, axis=1, sigma=np.exp(1), same_output_shape=True
    )
    result2 = gaussian_kernel_1D(
        x=x, center_idx=2, axis=1, sigma=np.exp(1), same_output_shape=True
    )
    assert_almost_equal(
        np.array(
            [
                [0.2573151, 0.2753297, 0.2573151, 0.2100401],
                [0.2573151, 0.2753297, 0.2573151, 0.2100401],
            ]
        ),
        result1,
    )
    assert_almost_equal(
        np.array(
            [
                [0.2100401, 0.2573151, 0.2753297, 0.2573151],
                [0.2100401, 0.2573151, 0.2753297, 0.2573151],
            ]
        ),
        result2,
    )
    assert np.shape(result1) == (2, 4)
    assert np.shape(result2) == (2, 4)
