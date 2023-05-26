import numpy as np
import pytest
import xarray as xr

from numpy.testing import assert_almost_equal

from kalman_reconstruction.pipeline import multiple_runs_of_func


def simple_function_01(ds, var="var", var_new="var_new"):
    """Multiply the variable by 2."""
    ds[var_new] = ds[var] * 2
    return ds


def simple_function_02(
    ds: xr.Dataset, var: str = "var", var_new: str = "var_new", value: int = 10
) -> xr.Dataset:
    """
    Perform a simple function on a dataset.

    This function creates a new dataset and performs a simple operation on a variable in the input dataset.
    It adds a new variable to the resulting dataset and assigns the sum of the input variable and a specified value to it.

    Parameters:
        ds (xr.Dataset): The input dataset.
        var (str): The name of the variable to operate on. Default is "var1".
        var_new (str): The name of the new variable to be added. Default is "var_new".
        value (int): The value to be added to the input variable. Default is 10.

    Returns:
        xr.Dataset: The resulting dataset with the modified variable.
    """
    # Create a new dataset with the same coordinates as the input dataset
    result = xr.Dataset(coords=ds.coords)

    # Assign additional coordinates to the resulting dataset
    result.assign_coords(dict(test=[0, 1]))

    # Copy the variable from the input dataset to the resulting dataset
    result[var] = ds[var]

    # Create a new DataArray with the specified value and coordinates from the resulting dataset
    result[var_new] = xr.DataArray(value, coords=result.coords)

    # Add the input variable and the new DataArray element-wise and assign the result to the new variable
    result[var_new] = result[var_new] + ds[var]

    return result


@pytest.fixture
def example_dataset_01():
    x = np.array([2, 3])
    y = np.array([0, 1, 2, 3])
    data_01 = np.array(
        [
            [0, 2, 3, np.nan],
            [1, 1, 1, 1],
        ]
    )
    return xr.Dataset(
        data_vars=dict(
            var=(("x", "y"), data_01),
        ),
        coords=dict(
            x=x,
            y=y,
        ),
    )


@pytest.fixture
def example_dataset_02():
    y = np.array([0, 1, 2, 3])
    z = np.array([7, 8])
    data_02 = np.array(
        [
            [0, 2],
            [1, 1],
            [2, np.nan],
            [0, 0],
        ]
    )
    return xr.Dataset(
        data_vars=dict(
            var=(("y", "z"), data_02),
        ),
        coords=dict(
            y=y,
            z=z,
        ),
    )


# @pytest.mark.parametrize("example_dataset", ["example_dataset_01", "example_dataset_02"])
# def test_multiple_runs_of_func(example_dataset):
#     ds = example_dataset  # Evaluate the example dataset fixture name

#     # Call the multiple_runs_of_func function
#     result = multiple_runs_of_func(ds=ds, func=simple_function_01, func_kwargs={}, number_of_runs=3, new_dimension="run")

#     # Perform assertions
#     assert isinstance(result, xr.Dataset)
#     assert "run" in result.dims
#     assert result.dims["run"] == 3
#     assert "var" in result.data_vars
#     assert "var_new" in result.data_vars
#     assert result.data_vars["var_new"].shape[0] == 3
#     assert result.data_vars["var_new"].shape[1:] == simple_function_01(ds=ds).data_vars["var_new"].shape

# def test_multiple_runs_of_func():

#     ds = example_dataset_02()
#     # Call the multiple_runs_of_func function
#     result = multiple_runs_of_func(ds=ds, func=simple_function_01, func_kwargs={}, number_of_runs=3, new_dimension="run")

#     # Perform assertions
#     assert isinstance(result, xr.Dataset)
#     assert "run" in result.dims
#     assert result.dims["run"] == 3
#     assert "var" in result.data_vars
#     assert "var_new" in result.data_vars
#     assert result.data_vars["var_new"].shape[0] == 3
#     assert result.data_vars["var_new"].shape[1:] == simple_function_01(ds=ds).data_vars["var_new"].shape
